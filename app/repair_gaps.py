"""Repair short, all-zero sample gaps in a WAV with LPC extrapolation.

Every gap is a half-open [start, end) interval of input sample indices that
must be 1-20 ms long, at least 40 ms away from either file edge, at least
60 ms away from any neighbouring gap, and contain zero samples on every
channel. Per channel a regularized 24th-order one-step linear predictor is
fit on the 30 ms of *original* samples on each side of the gap (ridge
least squares). The left predictor extrapolates forward into the gap, the
right predictor is fit on the reversed context and extrapolates backward;
the two estimates are linearly cross-faded by gap position. If either
context is all silence the request fails (422); if a prediction yields
non-finite values or anything outside the PCM full-scale range, that
channel falls back to the straight line through the two boundary samples.
Only samples inside the gaps are replaced, so the frame count, channel
format and every other PCM sample stay identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .loudness import (
    SAMPLE_RATE,
    AudioValidationError,
    _quantize,
    _read_wav,
    _write_wav,
)

# 48 kHz -> 48 frames per millisecond.
FRAMES_PER_MS = SAMPLE_RATE // 1000
MIN_GAP_FRAMES = 1 * FRAMES_PER_MS          # gap length: 1-20 ms
MAX_GAP_FRAMES = 20 * FRAMES_PER_MS
EDGE_GUARD_FRAMES = 40 * FRAMES_PER_MS      # >= 40 ms from head/tail
GAP_SEPARATION_FRAMES = 60 * FRAMES_PER_MS  # >= 60 ms between gaps
CONTEXT_FRAMES = 30 * FRAMES_PER_MS         # 30 ms context per side
PRED_ORDER = 24
# Ridge penalty as a fraction of the mean Gram diagonal (mean signal
# energy): scale-free and large enough to keep the 24x24 system SPD.
RIDGE_SCALE = 1e-3
FULL_SCALE = 1.0

RepairMethod = Literal["predicted", "linear"]


@dataclass
class GapRepair:
    start: int
    end: int
    method: RepairMethod


@dataclass
class RepairGapsResult:
    wav_bytes: bytes
    repairs: list[GapRepair]


def _validate_gaps(
    gaps: list[tuple[int, int]], n_frames: int
) -> list[tuple[int, int]]:
    """Check the half-open gap intervals and return them as plain ints.

    Gaps must be strictly ascending, 1-20 ms long, at least 40 ms from
    either edge and separated by at least 60 ms.
    """
    validated: list[tuple[int, int]] = []
    prev_end: int | None = None
    for start, end in gaps:
        if not 0 <= start < end <= n_frames:
            raise AudioValidationError(
                f"Gap [{start}, {end}) is empty or outside [0, {n_frames})"
            )
        length = end - start
        if not MIN_GAP_FRAMES <= length <= MAX_GAP_FRAMES:
            raise AudioValidationError(
                f"Gap [{start}, {end}) is {length} frames long; gaps must be "
                f"{MIN_GAP_FRAMES}-{MAX_GAP_FRAMES} frames (1-20 ms)"
            )
        if start < EDGE_GUARD_FRAMES or n_frames - end < EDGE_GUARD_FRAMES:
            raise AudioValidationError(
                f"Gap [{start}, {end}) must be at least "
                f"{EDGE_GUARD_FRAMES} frames (40 ms) from either file edge"
            )
        if prev_end is not None and start - prev_end < GAP_SEPARATION_FRAMES:
            raise AudioValidationError(
                "Consecutive gaps must be separated by at least "
                f"{GAP_SEPARATION_FRAMES} frames (60 ms)"
            )
        validated.append((start, end))
        prev_end = end
    return validated


def _fit_predictor(context: np.ndarray) -> np.ndarray:
    """Regularized forward one-step LPC coefficients from a 1-D context.

    Fits x[n] = sum_k a[k] * x[n-order+k], k = 0 .. PRED_ORDER-1, by
    ridge least squares over the context. Returns coefficients ordered
    oldest -> newest, matching the ascending history in _extrapolate.
    """
    order = PRED_ORDER
    # Rows are x[n-order:n] (ascending) for n = order .. m-1, targets x[n].
    windows = np.lib.stride_tricks.sliding_window_view(context, order)[:-1]
    targets = context[order:]
    gram = windows.T @ windows
    ridge = RIDGE_SCALE * float(np.trace(gram)) / order
    # Coefficients stay in ascending-history order (oldest -> newest) so
    # they line up directly with the ascending history used in _extrapolate.
    return np.linalg.solve(gram + ridge * np.eye(order), windows.T @ targets)


def _extrapolate(context: np.ndarray, weights: np.ndarray, length: int) -> np.ndarray:
    """Recursively extend `context` forward by `length` predicted samples."""
    order = weights.shape[0]
    history = context[-order:].astype(np.float64).copy()
    out = np.empty(length, dtype=np.float64)
    for i in range(length):
        value = float(np.dot(weights, history))
        out[i] = value
        history[:-1] = history[1:]
        history[-1] = value
    return out


def _linear_fill(channel: np.ndarray, start: int, end: int) -> np.ndarray:
    """Straight line through the boundary samples x[start-1], x[end]."""
    length = end - start
    left_value = float(channel[start - 1])
    right_value = float(channel[end])
    positions = (np.arange(length, dtype=np.float64) + 1.0) / (length + 1)
    return left_value + (right_value - left_value) * positions


def _repair_channel(
    channel: np.ndarray, start: int, end: int
) -> tuple[np.ndarray, bool]:
    """Fill one channel's gap; return (samples, used_prediction)."""
    length = end - start
    left = channel[start - CONTEXT_FRAMES : start]
    right = channel[end : end + CONTEXT_FRAMES]
    if not np.any(left) or not np.any(right):
        raise AudioValidationError(
            f"Gap [{start}, {end}) has an all-silence 30 ms context on at "
            "least one channel; prediction is impossible"
        )

    try:
        forward = _extrapolate(left, _fit_predictor(left), length)
        # Fit/extrapolate on the reversed right context, then reverse back
        # so indices line up with gap positions start .. end-1.
        backward = _extrapolate(
            right[::-1], _fit_predictor(right[::-1]), length
        )[::-1]
        if length == 1:
            blended = 0.5 * (forward + backward)
        else:
            right_weight = np.arange(length, dtype=np.float64) / (length - 1)
            blended = (1.0 - right_weight) * forward + right_weight * backward
        usable = (
            np.all(np.isfinite(forward))
            and np.all(np.isfinite(backward))
            and np.all(np.isfinite(blended))
            and np.max(np.abs(forward)) <= FULL_SCALE
            and np.max(np.abs(backward)) <= FULL_SCALE
            and np.max(np.abs(blended)) <= FULL_SCALE
        )
    except (np.linalg.LinAlgError, FloatingPointError):
        usable = False

    if usable:
        return blended, True
    return _linear_fill(channel, start, end), False


def repair_gaps_wav(
    wav_bytes: bytes, gaps: list[tuple[int, int]]
) -> RepairGapsResult:
    samples = _read_wav(wav_bytes)
    n_frames, channels = samples.shape

    validated = _validate_gaps(gaps, n_frames)
    repaired = samples.copy()
    repairs: list[GapRepair] = []
    for start, end in validated:
        if np.any(samples[start:end] != 0.0):
            raise AudioValidationError(
                f"Gap [{start}, {end}) contains non-zero samples on at "
                "least one channel"
            )

        fills: list[np.ndarray] = []
        predicted_flags: list[bool] = []
        for ch in range(channels):
            fill, predicted = _repair_channel(samples[:, ch], start, end)
            fills.append(fill)
            predicted_flags.append(predicted)
        # One label per gap: predicted only when every channel predicted.
        method: RepairMethod = "predicted" if all(predicted_flags) else "linear"
        repaired[start:end, :] = np.stack(fills, axis=1)
        repairs.append(GapRepair(start=start, end=end, method=method))

    pcm = _quantize(repaired)
    return RepairGapsResult(
        wav_bytes=_write_wav(pcm, channels), repairs=repairs
    )
