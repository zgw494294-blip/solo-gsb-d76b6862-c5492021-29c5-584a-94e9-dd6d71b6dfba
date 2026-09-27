"""HTTP-level concurrency tests: saturation, disconnects, recovery.

The audio computation is replaced by ``gated_normalize``, which runs in the
worker process and parks on the module-level ``GATE`` (inherited via fork)
until the test opens it — so tests control exactly when a running
computation finishes.

Two transports are exercised:

* ``httpx.ASGITransport`` with a cancelled request task — the in-process
  analogue of a disconnect (cancels the ASGI handler mid-await);
* a **real Uvicorn server on a real TCP socket**, where the client actually
  closes the TCP connection. That path is what triggers the ASGI
  ``http.disconnect`` event in production and is what the handler must watch.
"""
import asyncio
import base64
import http.client
import io
import json
import multiprocessing
import socket
import threading
import time
import urllib.error
import urllib.request
import wave

import httpx
import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient

import app.main as main
from app.executor import BoundedProcessExecutor
from app.loudness import normalize_wav as real_normalize_wav

RATE = 48000

# Module-level so forked worker processes inherit the same primitives.
GATE = multiprocessing.Event()
STARTED = multiprocessing.Value("i", 0)


def gated_normalize(wav_bytes, target_lufs, ceiling_dbtp):
    """normalize_wav that blocks at a test-controlled gate (runs in worker)."""
    with STARTED.get_lock():
        STARTED.value += 1
    GATE.wait(30)
    return real_normalize_wav(wav_bytes, target_lufs, ceiling_dbtp)


def make_payload(silence=False):
    t = np.arange(RATE * 2) / RATE
    sig = np.zeros_like(t) if silence else 0.1 * np.sin(2 * np.pi * 997 * t)
    pcm = np.clip(np.round(sig * 32768), -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(pcm.tobytes())
    return {
        "wav_base64": base64.b64encode(buf.getvalue()).decode(),
        "target_lufs": -23.0,
        "ceiling_dbtp": -1.0,
    }


@pytest.fixture
def tiny_executor(monkeypatch):
    """One worker + one queue slot, computation blocked at the test gate."""
    GATE.clear()
    with STARTED.get_lock():
        STARTED.value = 0
    monkeypatch.setattr(main, "normalize_wav", gated_normalize)
    executor = BoundedProcessExecutor(workers=1, queue_size=1)
    old = getattr(main.app.state, "executor", None)
    main.app.state.executor = executor
    yield executor
    main.app.state.executor = old
    GATE.set()  # unblock workers before joining the pool
    executor.shutdown()


async def wait_for(pred, timeout=10.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not pred():
        assert loop.time() < deadline, "condition not met in time"
        await asyncio.sleep(0.01)


def test_full_queue_503_and_health_bypass(tiny_executor):
    async def scenario():
        ex = tiny_executor
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            payload = make_payload()
            running = asyncio.create_task(
                client.post("/v1/normalize", json=payload)
            )
            await wait_for(lambda: STARTED.value >= 1)
            queued = asyncio.create_task(
                client.post("/v1/normalize", json=payload)
            )
            await wait_for(lambda: ex.queued == 1)

            # Worker busy + queue full: immediate 503 with Retry-After.
            resp = await client.post("/v1/normalize", json=payload)
            assert resp.status_code == 503
            assert resp.headers["retry-after"] == "1"

            # /health does not go through the queue.
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}

            GATE.set()
            assert (await running).status_code == 200
            assert (await queued).status_code == 200
            await wait_for(lambda: ex.running == 0 and ex.queued == 0)

    asyncio.run(scenario())


def test_two_phase_disconnect_and_recovery(tiny_executor):
    async def scenario():
        ex = tiny_executor
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            payload = make_payload()
            running = asyncio.create_task(
                client.post("/v1/normalize", json=payload)
            )
            await wait_for(lambda: STARTED.value >= 1)
            queued = asyncio.create_task(client.post("/v1/normalize", json=payload))
            await wait_for(lambda: ex.queued == 1)

            # Phase 1: the queued request disconnects — its slot is freed.
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            await wait_for(lambda: ex.queued == 0)
            admitted = asyncio.create_task(
                client.post("/v1/normalize", json=payload)
            )
            await wait_for(lambda: ex.queued == 1)

            # Phase 2: the running request disconnects — the worker slot is
            # held until the computation finishes, only the result is dropped.
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
            assert ex.running == 1
            assert STARTED.value == 1  # the admitted request has not started
            resp = await client.post("/v1/normalize", json=payload)
            assert resp.status_code == 503  # still saturated

            # Open the gate: the orphaned computation finishes and frees its
            # slot, then the queued request runs and gets the real result.
            GATE.set()
            resp = await admitted
            assert resp.status_code == 200
            data = resp.json()
            assert data["target_reached"] is True
            assert data["wav_base64"]

            # Capacity recovered: a new request is served normally.
            resp = await client.post("/v1/normalize", json=payload)
            assert resp.status_code == 200
            await wait_for(lambda: ex.running == 0 and ex.queued == 0)

    asyncio.run(scenario())


def test_business_error_releases_capacity(tiny_executor):
    async def scenario():
        ex = tiny_executor
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Silent WAV fails validation inside the worker (business error).
            failing = asyncio.create_task(
                client.post("/v1/normalize", json=make_payload(silence=True))
            )
            await wait_for(lambda: STARTED.value >= 1)
            queued = asyncio.create_task(
                client.post("/v1/normalize", json=make_payload())
            )
            await wait_for(lambda: ex.queued == 1)
            GATE.set()
            resp = await failing
            assert resp.status_code == 422  # original validation preserved
            # The queued request was promoted after the failure and succeeds.
            assert (await queued).status_code == 200
            await wait_for(lambda: ex.running == 0 and ex.queued == 0)
            resp = await client.post("/v1/normalize", json=make_payload())
            assert resp.status_code == 200

    asyncio.run(scenario())


def test_executor_created_and_reclaimed_with_lifespan():
    old = getattr(main.app.state, "executor", None)
    with TestClient(main.app) as client:
        executor = main.app.state.executor
        assert isinstance(executor, BoundedProcessExecutor)
        assert not executor.closed
        assert client.get("/health").status_code == 200
        resp = client.post("/v1/normalize", json=make_payload())
        assert resp.status_code == 200
        assert resp.json()["target_reached"] is True
    assert executor.closed
    assert getattr(main.app.state, "executor", None) is None
    if old is not None and old is not executor:
        old.shutdown()


# ---------------------------------------------------------------------------
# Real TCP socket + real Uvicorn: the client truly closes its connection, so
# the ASGI server emits http.disconnect (unlike ASGITransport, which only
# simulates it through task cancellation).
# ---------------------------------------------------------------------------


def _wait_until(pred, timeout=15.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(interval)


def _raw_post(host, port, payload):
    """Send one HTTP/1.1 request on a fresh socket; return the open socket."""
    body = json.dumps(payload).encode()
    request = (
        b"POST /v1/normalize HTTP/1.1\r\n"
        b"Host: %b:%d\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: %d\r\n"
        b"Connection: close\r\n"
        b"\r\n" % (host.encode(), port, len(body))
    ) + body
    sock = socket.create_connection((host, port), timeout=15)
    sock.sendall(request)
    return sock


def _read_response_status(sock):
    """Parse just the status line/body via http.client's HTTPResponse."""
    conn = http.client.HTTPResponse(sock, method="POST")
    conn.begin()
    body = conn.read()
    return conn.status, body


def _http_post_json(url, payload):
    """Plain stdlib POST; returns (status, body_bytes)."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


@pytest.fixture
def uvicorn_server(monkeypatch):
    """Real single-process Uvicorn on 127.0.0.1 with a gated 1+1 executor."""
    GATE.clear()
    with STARTED.get_lock():
        STARTED.value = 0
    monkeypatch.setattr(main, "normalize_wav", gated_normalize)
    monkeypatch.setenv("AUDIO_WORKERS", "1")
    monkeypatch.setenv("AUDIO_QUEUE", "1")
    # Replace the lazily-created/lifespan executor with the gated one by
    # forcing the lifespan to pick the env config; we track it via the
    # global app.state after startup.
    old = getattr(main.app.state, "executor", None)
    main.app.state.executor = None

    config = uvicorn.Config(
        main.app,
        host="127.0.0.1",
        port=0,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_until(lambda: server.started)
    host, port = server.servers[0].sockets[0].getsockname()[:2]
    executor = main.app.state.executor
    try:
        yield host, port, executor
    finally:
        GATE.set()  # unblock any orphaned worker before shutdown
        server.should_exit = True
        thread.join(timeout=15)
        main.app.state.executor = old


def test_real_socket_two_phase_disconnect_and_recovery(uvicorn_server):
    host, port, ex = uvicorn_server
    base_url = f"http://{host}:{port}"
    payload = make_payload()

    # Fill the single worker ...
    running_sock = _raw_post(host, port, payload)
    _wait_until(lambda: STARTED.value >= 1)
    # ... and the single FIFO queue slot.
    queued_sock = _raw_post(host, port, payload)
    _wait_until(lambda: ex.queued == 1)

    # Saturated: a third request over a real connection gets the 503.
    status, body = _http_post_json(f"{base_url}/v1/normalize", payload)
    assert status == 503
    assert json.loads(body)["detail"].startswith("Audio processing")

    # /health stays available while saturated.
    with urllib.request.urlopen(f"{base_url}/health", timeout=15) as resp:
        assert resp.status == 200
        assert json.loads(resp.read()) == {"status": "ok"}

    # Phase 1: the QUEUED client closes its TCP connection. The handler must
    # see http.disconnect, cancel the queued wait and free the queue slot.
    queued_sock.close()
    _wait_until(lambda: ex.queued == 0)
    # A new request takes the freed slot immediately — previously it would
    # have received a spurious 503 because the dead waiter held the slot.
    admitted_sock = _raw_post(host, port, payload)
    _wait_until(lambda: ex.queued == 1)

    # Phase 2: the RUNNING client closes its TCP connection. The worker slot
    # stays occupied until the computation ends; only the result is dropped.
    running_sock.close()
    # Give the server a moment to observe the close, then assert the slot is
    # still held (a premature release would promote the admitted request).
    time.sleep(0.3)
    assert ex.running == 1
    _wait_until(lambda: STARTED.value == 1, timeout=2.0)
    # Still saturated: a new request is rejected with 503.
    status, _ = _http_post_json(f"{base_url}/v1/normalize", payload)
    assert status == 503

    # Open the gate: the orphaned computation finishes and frees the worker,
    # then the admitted (still connected) request runs FIFO and gets 200.
    GATE.set()
    status, body = _read_response_status(admitted_sock)
    assert status == 200
    data = json.loads(body)
    assert data["target_reached"] is True
    assert data["wav_base64"]
    _wait_until(lambda: ex.running == 0 and ex.queued == 0)

    # Capacity fully recovered: a normal request over a real socket works and
    # the original response shape is preserved.
    status, body = _http_post_json(f"{base_url}/v1/normalize", payload)
    assert status == 200
    data = json.loads(body)
    assert data["target_reached"] is True
    assert data["wav_base64"]
    assert "gain_db" in data and "output_loudness_lufs" in data


def test_real_socket_disconnect_before_computation_finishes_then_recovers(
    uvicorn_server,
):
    """Queued disconnect: the freed queue slot admits a request that is later
    served with the real result after the (un-interrupted) running job ends."""
    host, port, ex = uvicorn_server
    base_url = f"http://{host}:{port}"
    payload = make_payload()

    running_sock = _raw_post(host, port, payload)
    _wait_until(lambda: STARTED.value >= 1)
    queued_sock = _raw_post(host, port, payload)
    _wait_until(lambda: ex.queued == 1)

    # Disconnect the queued client and replace it with a live waiter.
    queued_sock.close()
    _wait_until(lambda: ex.queued == 0)
    admitted_sock = _raw_post(host, port, payload)
    _wait_until(lambda: ex.queued == 1)

    # Finish the running job normally: its own client still receives 200,
    # then the admitted waiter is promoted FIFO and completes.
    GATE.set()
    assert _read_response_status(running_sock)[0] == 200
    status, _ = _read_response_status(admitted_sock)
    assert status == 200
    _wait_until(lambda: ex.running == 0 and ex.queued == 0)

    # And capacity is reusable.
    with urllib.request.urlopen(f"{base_url}/health", timeout=15) as resp:
        assert resp.status == 200
