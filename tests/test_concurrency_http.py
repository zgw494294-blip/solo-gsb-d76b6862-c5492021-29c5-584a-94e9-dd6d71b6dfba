"""HTTP-level concurrency tests: saturation, disconnects, recovery.

The audio computation is replaced by ``gated_normalize``, which runs in the
worker process and parks on the module-level ``GATE`` (inherited via fork)
until the test opens it — so tests control exactly when a running
computation finishes. Cancelling the httpx request task cancels the ASGI
handler mid-await, which is how uvicorn reports a client disconnect.
"""
import asyncio
import base64
import io
import multiprocessing
import wave

import httpx
import numpy as np
import pytest
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
