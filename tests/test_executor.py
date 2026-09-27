"""Concurrency tests for the shared bounded process executor.

The blocking points of the computation are controlled from the test process:
``gated_value`` runs in a worker process and parks on the module-level
``GATE`` (a ``multiprocessing.Event`` inherited via fork) until the test
opens it, so tests decide exactly when a running computation finishes.
"""
import asyncio
import multiprocessing
import queue as queue_module

import pytest

from app.executor import (
    BoundedProcessExecutor,
    ExecutorClosedError,
    ExecutorOverloadedError,
)
from app.loudness import AudioValidationError

# Module-level so forked worker processes inherit the same underlying
# semaphores no matter when the pool spawns its workers.
GATE = multiprocessing.Event()
STARTED = multiprocessing.Value("i", 0)
ORDER = multiprocessing.Queue()


def gated_value(tag):
    """Block at the test-controlled gate, then return ``tag``."""
    with STARTED.get_lock():
        STARTED.value += 1
    ORDER.put(("start", tag))
    GATE.wait(30)
    ORDER.put(("end", tag))
    return tag


def quick_value(tag):
    return tag


def boom():
    raise AudioValidationError("bad wav")


def gated_boom():
    """Block at the test-controlled gate, then raise a business error."""
    with STARTED.get_lock():
        STARTED.value += 1
    GATE.wait(30)
    raise AudioValidationError("bad wav")


@pytest.fixture(autouse=True)
def clean_gate():
    GATE.clear()
    with STARTED.get_lock():
        STARTED.value = 0
    while True:
        try:
            ORDER.get_nowait()
        except queue_module.Empty:
            break
    yield
    GATE.set()  # never leave a worker blocked at teardown


async def wait_for(pred, timeout=10.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not pred():
        assert loop.time() < deadline, "condition not met in time"
        await asyncio.sleep(0.01)


def wait_started(count):
    return wait_for(lambda: STARTED.value >= count)


def drain_order(count):
    return [ORDER.get(timeout=10) for _ in range(count)]


def test_result_and_business_exception_passthrough(clean_gate):
    async def main():
        ex = BoundedProcessExecutor(workers=1, queue_size=1)
        try:
            assert await ex.run(quick_value, "ok") == "ok"
            with pytest.raises(AudioValidationError, match="bad wav"):
                await ex.run(boom)
            assert ex.running == 0 and ex.queued == 0
        finally:
            ex.shutdown()

    asyncio.run(main())


def test_full_queue_rejected_without_piling_up(clean_gate):
    async def main():
        ex = BoundedProcessExecutor(workers=1, queue_size=1)
        try:
            t1 = asyncio.create_task(ex.run(gated_value, "a"))
            await wait_started(1)
            t2 = asyncio.create_task(ex.run(gated_value, "b"))
            await wait_for(lambda: ex.queued == 1)
            # Worker busy + queue full: rejected immediately, and nothing
            # extra is handed to the process pool.
            with pytest.raises(ExecutorOverloadedError):
                await ex.run(gated_value, "c")
            assert ex.running == 1 and ex.queued == 1
            GATE.set()
            assert await t1 == "a"
            assert await t2 == "b"
            assert ex.running == 0 and ex.queued == 0
        finally:
            ex.shutdown()

    asyncio.run(main())


def test_zero_queue_accepts_only_workers(clean_gate):
    async def main():
        ex = BoundedProcessExecutor(workers=1, queue_size=0)
        try:
            t1 = asyncio.create_task(ex.run(gated_value, "a"))
            await wait_started(1)
            with pytest.raises(ExecutorOverloadedError):
                await ex.run(gated_value, "b")
            GATE.set()
            assert await t1 == "a"
        finally:
            ex.shutdown()

    asyncio.run(main())


def test_queued_disconnect_frees_slot_and_no_queue_jumping(clean_gate):
    async def main():
        ex = BoundedProcessExecutor(workers=1, queue_size=2)
        try:
            t1 = asyncio.create_task(ex.run(gated_value, "a"))
            await wait_started(1)
            t2 = asyncio.create_task(ex.run(gated_value, "b"))
            t3 = asyncio.create_task(ex.run(gated_value, "c"))
            await wait_for(lambda: ex.queued == 2)
            # Disconnect while queued: t2 is removed and its slot freed.
            t2.cancel()
            with pytest.raises(asyncio.CancelledError):
                await t2
            await wait_for(lambda: ex.queued == 1)
            # A later request takes the freed slot but must not jump ahead
            # of t3, which was already waiting.
            t4 = asyncio.create_task(ex.run(gated_value, "d"))
            await wait_for(lambda: ex.queued == 2)
            GATE.set()
            assert await t1 == "a"
            assert await t3 == "c"
            assert await t4 == "d"
            assert drain_order(6) == [
                ("start", "a"),
                ("end", "a"),
                ("start", "c"),
                ("end", "c"),
                ("start", "d"),
                ("end", "d"),
            ]
            assert ex.running == 0 and ex.queued == 0
        finally:
            ex.shutdown()

    asyncio.run(main())


def test_running_disconnect_holds_slot_until_computation_ends(clean_gate):
    async def main():
        ex = BoundedProcessExecutor(workers=1, queue_size=1)
        try:
            t1 = asyncio.create_task(ex.run(gated_value, "a"))
            await wait_started(1)
            # Disconnect while running: the waiter is cancelled but the
            # worker slot stays taken until the computation finishes.
            t1.cancel()
            with pytest.raises(asyncio.CancelledError):
                await t1
            assert ex.running == 1
            t2 = asyncio.create_task(ex.run(gated_value, "b"))
            await wait_for(lambda: ex.queued == 1)
            # t1's slot is not reusable yet and t2 holds the queue slot.
            with pytest.raises(ExecutorOverloadedError):
                await ex.run(gated_value, "c")
            assert STARTED.value == 1  # t2 has not started early
            GATE.set()
            assert await t2 == "b"
            await wait_for(lambda: ex.running == 0 and ex.queued == 0)
            # t2 started only after t1's computation actually finished.
            assert drain_order(4) == [
                ("start", "a"),
                ("end", "a"),
                ("start", "b"),
                ("end", "b"),
            ]
        finally:
            ex.shutdown()

    asyncio.run(main())


def test_business_exception_releases_capacity(clean_gate):
    async def main():
        ex = BoundedProcessExecutor(workers=1, queue_size=1)
        try:
            t1 = asyncio.create_task(ex.run(gated_boom))
            await wait_started(1)
            t2 = asyncio.create_task(ex.run(gated_value, "b"))
            await wait_for(lambda: ex.queued == 1)
            GATE.set()
            with pytest.raises(AudioValidationError):
                await t1
            # The queued job was promoted after the failure and completes.
            assert await t2 == "b"
            await wait_for(lambda: ex.running == 0 and ex.queued == 0)
            # Capacity is fully usable again.
            assert await ex.run(quick_value, "z") == "z"
        finally:
            ex.shutdown()

    asyncio.run(main())


def test_cancel_complete_race_releases_exactly_once(clean_gate):
    async def main():
        ex = BoundedProcessExecutor(workers=1, queue_size=1)
        try:
            t1 = asyncio.create_task(ex.run(gated_value, "a"))
            await wait_started(1)
            t2 = asyncio.create_task(ex.run(gated_value, "b"))
            await wait_for(lambda: ex.queued == 1)
            # Cancel the queued waiter at the same time as the running
            # computation completes: exactly one slot may be freed.
            t2.cancel()
            GATE.set()
            with pytest.raises(asyncio.CancelledError):
                await t2
            assert await t1 == "a"
            await wait_for(lambda: ex.running == 0 and ex.queued == 0)
            # Accounting is still exact: one running + one queued, and a
            # third is rejected (a double release would admit it).
            GATE.clear()
            t3 = asyncio.create_task(ex.run(gated_value, "c"))
            await wait_started(2)
            t4 = asyncio.create_task(ex.run(gated_value, "d"))
            await wait_for(lambda: ex.queued == 1)
            with pytest.raises(ExecutorOverloadedError):
                await ex.run(gated_value, "e")
            GATE.set()
            assert await t3 == "c"
            assert await t4 == "d"
            await wait_for(lambda: ex.running == 0 and ex.queued == 0)
        finally:
            ex.shutdown()

    asyncio.run(main())


def test_shutdown_fails_queued_waiters_and_closes(clean_gate):
    async def main():
        ex = BoundedProcessExecutor(workers=1, queue_size=1)
        t1 = asyncio.create_task(ex.run(gated_value, "a"))
        await wait_started(1)
        t2 = asyncio.create_task(ex.run(quick_value, "b"))
        await wait_for(lambda: ex.queued == 1)
        shutdown_task = asyncio.create_task(asyncio.to_thread(ex.shutdown))
        with pytest.raises(ExecutorClosedError):
            await t2
        GATE.set()
        assert await t1 == "a"  # running jobs finish before shutdown returns
        await shutdown_task
        assert ex.closed
        with pytest.raises(ExecutorClosedError):
            await ex.run(quick_value, "c")

    asyncio.run(main())


def test_env_configuration(monkeypatch):
    monkeypatch.setenv("AUDIO_WORKERS", "3")
    monkeypatch.setenv("AUDIO_QUEUE", "0")
    ex = BoundedProcessExecutor.from_env()
    try:
        assert (ex.workers, ex.queue_size) == (3, 0)
    finally:
        ex.shutdown()

    monkeypatch.setenv("AUDIO_WORKERS", "99")
    monkeypatch.setenv("AUDIO_QUEUE", "-5")
    ex = BoundedProcessExecutor.from_env()
    try:
        assert (ex.workers, ex.queue_size) == (8, 0)  # clamped to range
    finally:
        ex.shutdown()

    monkeypatch.setenv("AUDIO_WORKERS", "not-a-number")
    monkeypatch.delenv("AUDIO_QUEUE")
    ex = BoundedProcessExecutor.from_env()
    try:
        assert (ex.workers, ex.queue_size) == (2, 4)  # defaults
    finally:
        ex.shutdown()
