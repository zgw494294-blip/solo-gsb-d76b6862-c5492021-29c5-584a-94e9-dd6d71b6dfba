"""Shared bounded process executor for audio computation.

Every audio POST endpoint hands its heavy computation to one executor: the
work runs in a pool of worker processes while the HTTP handler awaits the
result. At most ``AUDIO_WORKERS`` jobs run concurrently and up to
``AUDIO_QUEUE`` more wait in a FIFO queue; anything beyond that is rejected
immediately with :class:`ExecutorOverloadedError` (mapped to
``503 + Retry-After: 1`` by the app) instead of piling up inside the process
pool.

Disconnect semantics: a queued job whose waiter goes away is removed and its
queue slot is freed; a running job keeps its worker slot until the
computation actually finishes — only the result is discarded. Completion,
business exceptions and cancellation races each release a slot exactly once,
and queued jobs are promoted strictly in FIFO order (no queue jumping).
"""
from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import multiprocessing
import os
import threading
from typing import Any, Callable

DEFAULT_WORKERS = 2
DEFAULT_QUEUE = 4
MIN_WORKERS, MAX_WORKERS = 1, 8
MIN_QUEUE, MAX_QUEUE = 0, 32


class ExecutorOverloadedError(Exception):
    """No worker or queue slot is available (mapped to HTTP 503)."""


class ExecutorClosedError(ExecutorOverloadedError):
    """The executor is shut down and accepts no new jobs."""


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _mp_context() -> multiprocessing.context.BaseContext:
    # Fork so the worker processes inherit the already-imported app modules
    # (and, in tests, module-level synchronization primitives created before
    # the pool spawns its workers).
    try:
        return multiprocessing.get_context("fork")
    except ValueError:  # platforms without fork
        return multiprocessing.get_context()


class _Job:
    """One submitted computation and its state machine.

    States: ``queued`` -> ``running`` -> ``done``; a running job whose waiter
    is cancelled goes ``running`` -> ``cancelled`` -> ``done`` (the worker
    slot is released only when the computation finishes). A queued job whose
    waiter is cancelled goes straight to ``done`` and leaves the queue.
    """

    __slots__ = ("fn", "args", "loop", "aio_future", "cf_future", "state")

    def __init__(
        self,
        fn: Callable[..., Any],
        args: tuple,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.fn = fn
        self.args = args
        self.loop = loop
        self.aio_future: asyncio.Future = loop.create_future()
        self.cf_future: concurrent.futures.Future | None = None
        self.state = "queued"


class BoundedProcessExecutor:
    """FIFO-bounded wrapper around a ``ProcessPoolExecutor``."""

    def __init__(
        self, workers: int = DEFAULT_WORKERS, queue_size: int = DEFAULT_QUEUE
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be >= 1")
        if queue_size < 0:
            raise ValueError("queue_size must be >= 0")
        self._workers = workers
        self._queue_size = queue_size
        self._pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=_mp_context()
        )
        self._lock = threading.Lock()
        self._pending: collections.deque[_Job] = collections.deque()
        self._running = 0
        self._closed = False

    @classmethod
    def from_env(cls) -> BoundedProcessExecutor:
        return cls(
            workers=_env_int(
                "AUDIO_WORKERS", DEFAULT_WORKERS, MIN_WORKERS, MAX_WORKERS
            ),
            queue_size=_env_int(
                "AUDIO_QUEUE", DEFAULT_QUEUE, MIN_QUEUE, MAX_QUEUE
            ),
        )

    @property
    def workers(self) -> int:
        return self._workers

    @property
    def queue_size(self) -> int:
        return self._queue_size

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def running(self) -> int:
        with self._lock:
            return self._running

    @property
    def queued(self) -> int:
        with self._lock:
            return len(self._pending)

    async def run(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run ``fn(*args)`` in a worker process and await its result.

        Raises :class:`ExecutorOverloadedError` immediately when every worker
        and queue slot is taken; the job is never handed to the pool in that
        case. If the awaiting task is cancelled (client disconnect), a queued
        job is dropped and its slot freed; a running job is left to finish in
        the worker and only its result is discarded.
        """
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._closed:
                raise ExecutorClosedError("audio executor is shut down")
            if (
                self._running + len(self._pending)
                >= self._workers + self._queue_size
            ):
                raise ExecutorOverloadedError(
                    "audio executor is saturated: all worker and queue "
                    "slots are taken"
                )
            job = _Job(fn, args, loop)
            if self._running < self._workers:
                self._start_locked(job)
            else:
                self._pending.append(job)
        try:
            return await job.aio_future
        except asyncio.CancelledError:
            self._on_waiter_cancelled(job)
            raise

    def shutdown(self) -> None:
        """Stop accepting jobs, fail queued waiters, join the worker pool."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pending = list(self._pending)
            self._pending.clear()
            for job in pending:
                job.state = "done"
        for job in pending:
            self._fail_on_loop(job)
        self._pool.shutdown(wait=True)

    # -- internals (all state transitions guarded by ``self._lock``) --

    def _start_locked(self, job: _Job) -> None:
        # Submit first: if the pool refuses the job (shut down/broken) the
        # capacity accounting stays untouched.
        cf_future = self._pool.submit(job.fn, *job.args)
        job.state = "running"
        self._running += 1
        job.cf_future = cf_future
        cf_future.add_done_callback(
            lambda future: self._schedule_done(job, future)
        )

    def _schedule_done(
        self, job: _Job, cf_future: concurrent.futures.Future
    ) -> None:
        try:
            job.loop.call_soon_threadsafe(self._on_done, job, cf_future)
        except RuntimeError:
            # The submitter's loop is already closed (interpreter/test
            # teardown): settle inline from this pool thread.
            self._on_done(job, cf_future)

    def _on_done(self, job: _Job, cf_future: concurrent.futures.Future) -> None:
        with self._lock:
            if job.state == "done":
                return  # defensive: a slot is released exactly once
            was_cancelled = job.state == "cancelled"
            job.state = "done"
            self._running -= 1
            self._promote_locked()
        if was_cancelled:
            return  # waiter is gone: discard result and exception
        self._settle(job, cf_future)

    def _promote_locked(self) -> None:
        # Invariant: the queue is non-empty only when every worker is busy,
        # so this starts at most one job per freed slot, in FIFO order.
        while self._pending and self._running < self._workers:
            job = self._pending.popleft()
            try:
                self._start_locked(job)
            except RuntimeError as exc:
                # The pool is gone (shut down or broken): fail the waiter
                # instead of leaving it queued forever.
                job.state = "done"
                self._fail_on_loop(job, exc)

    def _settle(self, job: _Job, cf_future: concurrent.futures.Future) -> None:
        future = job.aio_future
        if future.done():
            return
        try:
            result = cf_future.result()
        except BaseException as exc:
            try:
                future.set_exception(exc)
            except (RuntimeError, asyncio.InvalidStateError):
                pass  # loop closed or waiter already gone
        else:
            try:
                future.set_result(result)
            except (RuntimeError, asyncio.InvalidStateError):
                pass

    def _on_waiter_cancelled(self, job: _Job) -> None:
        with self._lock:
            if job.state == "queued":
                # Disconnect while queued: leave the FIFO, free the slot.
                job.state = "done"
                self._pending.remove(job)
            elif job.state == "running":
                # Disconnect while running: keep the worker slot until the
                # computation finishes; _on_done will discard the result.
                job.state = "cancelled"
            # "done"/"cancelled": the slot was (or will be) released by the
            # completion path — never released twice.

    def _fail_on_loop(self, job: _Job, exc: BaseException | None = None) -> None:
        if exc is None:
            exc = ExecutorClosedError("audio executor shut down")

        def fail() -> None:
            future = job.aio_future
            if not future.done():
                try:
                    future.set_exception(exc)
                except (RuntimeError, asyncio.InvalidStateError):
                    pass

        try:
            job.loop.call_soon_threadsafe(fail)
        except RuntimeError:
            pass  # loop already closed
