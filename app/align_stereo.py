"""Stereo track alignment driven by a user-marked calibration interval.

``calibration_interval`` is a half-open [start, end) range of input sample
indices where the two tracks of a stereo WAV are known to carry the same
source. For every integer offset in -240..+240 frames the valid overlap of
the two tracks inside the interval is demeaned per track and scored by the
normalized cross-correlation; the offset with the largest absolute
correlation wins, with ties broken by the smallest absolute offset and
then the smallest numeric offset. ``offset_samples`` is the right-track
index minus the left-track index of corresponding samples, so a positive
value means the right track is delayed.

Every statistic is exact integer arithmetic on the 16-bit samples: with
n = overlap length the demeaned sums are scaled through by n,

    cross   = n*sum(L*R) - sum(L)*sum(R)
    e_left  = n*sum(L*L) - sum(L)*sum(L)
    e_right = n*sum(R*R) - sum(R)*sum(R)

and corr = cross / sqrt(e_left * e_right), so ties are genuine and the
0.8 acceptance threshold is the exact comparison
25*cross^2 >= 16*e_left*e_right. Offsets whose overlap has zero variance
on either track are invalid; no valid candidate at all, or a best absolute
correlation below 0.8, rejects the request.

The left track is passed through sample-for-sample. The whole right track
is shifted by the chosen offset (aligned[i] = right[i + offset],
out-of-range frames zero-filled) and, when the winning correlation is
negative, inverted by negating in 32-bit and clamping back to 16-bit, so
-32768 maps to 32767 instead of overflowing. A zero offset with positive
correlation returns the PCM unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .compare_gain import _corr_exceeds
from .loudness import SAMPLE_RATE, AudioValidationError, _read_wav, _write_wav

MAX_OFFSET_FRAMES = 240  # search range: -240..+240 frames
MIN_INTERVAL_FRAMES = SAMPLE_RATE // 5  # calibration length: 0.2-2 s
MAX_INTERVAL_FRAMES = 2 * SAMPLE_RATE


@dataclass
class AlignStereoResult:
    wav_bytes: bytes
    offset_samples: int
    polarity_inverted: bool
    correlation: float


def _validate_calibration_interval(
    calibration_interval: tuple[int, int], n_frames: int
) -> tuple[int, int]:
    """Check the half-open calibration interval and return it as plain ints.

    The interval must be non-empty, inside [0, n_frames) and 0.2-2 s long.
    """
    start, end = int(calibration_interval[0]), int(calibration_interval[1])
    if not 0 <= start < end <= n_frames:
        raise AudioValidationError(
            f"calibration_interval [{start}, {end}) is empty or outside "
            f"[0, {n_frames})"
        )
    length = end - start
    if not MIN_INTERVAL_FRAMES <= length <= MAX_INTERVAL_FRAMES:
        raise AudioValidationError(
            f"calibration_interval [{start}, {end}) is {length} frames "
            f"long; the interval must be {MIN_INTERVAL_FRAMES}-"
            f"{MAX_INTERVAL_FRAMES} frames (0.2-2 s)"
        )
    return start, end


def _int_cumsum(values: np.ndarray) -> np.ndarray:
    """Cumsum with a leading zero (int64, exact for 16-bit input)."""
    cs = np.zeros(values.shape[0] + 1, dtype=np.int64)
    np.cumsum(values, out=cs[1:])
    return cs


def _select_offset(
    left: np.ndarray, right: np.ndarray, start: int, end: int
) -> tuple[int, int, int, int]:
    """Pick the winning offset; return (offset, cross, e_left, e_right).

    ``left``/``right`` hold the 16-bit samples as int64. For offset d the
    overlap inside [start, end) pairs left[i] with right[i + d]; the
    demeaned normalized cross-correlation is computed with the exact
    integer statistics from the module docstring. Offsets are visited in
    tie-break order (ascending |d|, then ascending d) and the incumbent is
    only displaced by a strictly larger exact |correlation|, so ties keep
    the smallest absolute offset, then the smallest numeric offset.
    """
    length = end - start
    cs_l = _int_cumsum(left)
    cs_r = _int_cumsum(right)
    cs_l2 = _int_cumsum(left * left)
    cs_r2 = _int_cumsum(right * right)

    best_offset = 0
    best_stats: tuple[int, int, int] | None = None
    for offset in sorted(
        range(-MAX_OFFSET_FRAMES, MAX_OFFSET_FRAMES + 1),
        key=lambda d: (abs(d), d),
    ):
        n = length - abs(offset)
        l_lo = start - min(offset, 0)
        r_lo = l_lo + offset
        l_hi = l_lo + n
        r_hi = r_lo + n
        s_l = int(cs_l[l_hi]) - int(cs_l[l_lo])
        s_r = int(cs_r[r_hi]) - int(cs_r[r_lo])
        q_l = int(cs_l2[l_hi]) - int(cs_l2[l_lo])
        q_r = int(cs_r2[r_hi]) - int(cs_r2[r_lo])
        # sum(L*R) over the overlap: |each product| <= 2**30 and
        # n <= 96000, so the int64 dot product is exact.
        x = int(np.dot(left[l_lo:l_hi], right[r_lo:r_hi]))
        stats = (n * x - s_l * s_r, n * q_l - s_l * s_l, n * q_r - s_r * s_r)
        if stats[1] <= 0 or stats[2] <= 0:
            continue  # zero-variance overlap: correlation undefined
        if best_stats is None or _corr_exceeds(stats, best_stats):
            best_stats = stats
            best_offset = offset

    if best_stats is None:
        raise AudioValidationError(
            "No valid candidate offset: the calibration interval has a "
            "zero-variance overlap on every offset"
        )
    cross, e_left, e_right = best_stats
    # |corr| >= 0.8 exactly: 25*cross^2 >= 16*e_left*e_right.
    if 25 * cross * cross < 16 * e_left * e_right:
        raise AudioValidationError(
            "Best absolute correlation in the calibration interval is "
            "below 0.8"
        )
    return best_offset, cross, e_left, e_right


def _shift_track(track: np.ndarray, offset: int, invert: bool) -> np.ndarray:
    """aligned[i] = track[i + offset] (negated if invert), zero-filled.

    The negation is done in 32-bit and clamped back to 16-bit, so -32768
    becomes 32767 instead of wrapping around.
    """
    n_frames = track.shape[0]
    widened = track.astype(np.int32)
    if invert:
        widened = -widened
    aligned = np.zeros(n_frames, dtype=np.int32)
    lo = max(0, -offset)
    hi = min(n_frames, n_frames - offset)
    aligned[lo:hi] = widened[lo + offset : hi + offset]
    return np.clip(aligned, -32768, 32767).astype("<i2")


def align_stereo_wav(
    wav_bytes: bytes, calibration_interval: tuple[int, int]
) -> AlignStereoResult:
    samples = _read_wav(wav_bytes)
    n_frames, channels = samples.shape
    if channels != 2:
        raise AudioValidationError(
            f"WAV must be stereo (got {channels} channel(s))"
        )
    start, end = _validate_calibration_interval(calibration_interval, n_frames)

    # Decoded samples are exact int16/32768, so rint recovers the PCM
    # integers exactly.
    pcm = np.rint(samples * 32768.0).astype("<i2")
    left = pcm[:, 0].astype(np.int64)
    right = pcm[:, 1].astype(np.int64)

    offset, cross, e_left, e_right = _select_offset(left, right, start, end)
    inverted = cross < 0
    correlation = cross / math.sqrt(e_left * e_right)

    aligned = np.empty((n_frames, 2), dtype="<i2")
    aligned[:, 0] = pcm[:, 0]  # left track: sample-for-sample unchanged
    aligned[:, 1] = _shift_track(pcm[:, 1], offset, inverted)
    return AlignStereoResult(
        wav_bytes=_write_wav(aligned, 2),
        offset_samples=offset,
        polarity_inverted=inverted,
        correlation=correlation,
    )
