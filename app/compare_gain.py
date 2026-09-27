"""Gain/shift comparison between two compliant WAV files.

Given a reference and a candidate (same sample format / channel count),
check whether their overlapping content differs only by:

* one integer-sample time shift (within +/-100 ms), and
* one signed linear gain common to all channels.

The shift is found as the lag maximizing the overlap-energy-normalized
cross-correlation (pooled over channels). The exact winner is chosen with
integer arithmetic: the pooled cross-sum for *every* searched lag is
computed in one batched split-base FFT (each component convolution is
rounded to an integer under a provable round-off bound hundreds of times
smaller than 0.5; a lag that could not be certified falls back to a direct
integer dot product), and overlap energies come from integer cumulative
sums. Hence no FFT tolerance band is needed at all -- two lags compare
equal iff their normalized correlations are genuinely identical, and ties
go to the smallest absolute lag, then the smallest numeric lag. The gain
is a single least-squares scalar over the aligned overlap of every
channel. Residuals are then scanned in non-overlapping 100 ms windows on
the reference timeline (the tail window is included) and windows whose RMS
exceeds an absolute floor of 2/32768 or 1% of the reference overlap RMS
are merged into half-open anomaly intervals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .loudness import SAMPLE_RATE, AudioValidationError, _read_wav

MAX_SHIFT_MS = 100.0
WINDOW_MS = 100.0
MAX_SHIFT_SAMPLES = round(SAMPLE_RATE * MAX_SHIFT_MS / 1000.0)  # +/-4800
WINDOW_SAMPLES = round(SAMPLE_RATE * WINDOW_MS / 1000.0)  # 4800
RESIDUAL_FLOOR = 2.0 / 32768.0
RESIDUAL_RELATIVE = 0.01
# Split base for the exact batched cross-correlation: int16 sample x is
# written x = hi*SPLIT_BASE + lo with 0 <= lo < SPLIT_BASE (floor
# division, so this also holds for negative x).
SPLIT_BASE = 1 << 8
# Two 10 s files (959,999 linear-correlation samples) pad to a
# 2**20-point FFT.
_MAX_FFT_LOG2 = 20
# Log2 FFT size used by the round-off certificate; kept slightly above
# the hard size cap so the bound is a uniform, slightly conservative
# upper bound for every compliant input without depending on the
# actual transform length.
_CERT_FFT_LOG2 = 21


@dataclass
class CompareGainResult:
    offset_samples: int
    gain_db: float
    polarity_inverted: bool
    reference_unmatched_samples: int
    candidate_unmatched_samples: int
    # Half-open [start_sample, end_sample) intervals on the reference
    # timeline where residuals exceed the per-window threshold.
    anomaly_intervals: list[tuple[int, int]] = field(default_factory=list)
    overlap_gain_only: bool = True


def _overlap_slice(lengths: tuple[int, int], lag: int) -> tuple[int, int, int]:
    """Return (ref_start, cand_start, overlap_length) for lag d.

    Candidate frame (i + d) aligns with reference frame i, so the reported
    offset is candidate_index - reference_index.
    """
    n_ref, n_cand = lengths
    ref_start = max(-lag, 0)
    cand_start = max(lag, 0)
    overlap = min(n_ref - ref_start, n_cand - cand_start)
    return ref_start, cand_start, overlap


def _int_cumsum_energy(samples: np.ndarray) -> np.ndarray:
    """Pooled per-frame energy cumsum with a leading zero (int64)."""
    cs = np.zeros(samples.shape[0] + 1, dtype=np.int64)
    np.cumsum((samples * samples).sum(axis=1), out=cs[1:])
    return cs


def _exact_overlap_stats(
    ref_int: np.ndarray, cand_int: np.ndarray, lag: int
) -> tuple[int, int, int]:
    """Exact (cross_sum, ref_energy, cand_energy) over the overlap at lag.

    The inputs hold the 16-bit sample values as int64, so every product and
    partial sum is exact integer arithmetic: two lags compare equal iff
    their normalized correlations are genuinely identical.
    """
    r_lo, c_lo, length = _overlap_slice(
        (ref_int.shape[0], cand_int.shape[0]), lag
    )
    ref_seg = ref_int[r_lo : r_lo + length].reshape(-1)
    cand_seg = cand_int[c_lo : c_lo + length].reshape(-1)
    cross = int(np.dot(ref_seg, cand_seg))
    e_ref = int(np.dot(ref_seg, ref_seg))
    e_cand = int(np.dot(cand_seg, cand_seg))
    return cross, e_ref, e_cand


def _corr_exceeds(
    stats_a: tuple[int, int, int], stats_b: tuple[int, int, int]
) -> bool:
    """True if stats_a's |correlation| strictly exceeds stats_b's.

    Compares cross^2 / (e_ref * e_cand) across lags with exact integer
    arithmetic (Python ints, no overflow), so true ties stay tied.
    """
    x_a, e_ref_a, e_cand_a = stats_a
    x_b, e_ref_b, e_cand_b = stats_b
    return (
        x_a * x_a * e_ref_b * e_cand_b
        > x_b * x_b * e_ref_a * e_cand_a
    )


def _pad_signal(values: np.ndarray, fft_len: int, length: int) -> np.ndarray:
    """Zero-pad a real-valued column up to the FFT length."""
    out = np.zeros(fft_len, dtype=np.float64)
    out[:length] = values
    return out


def _roundoff_certified(
    values: np.ndarray, bound: float
) -> np.ndarray:
    """Boolean mask: lags whose FFT value rounds to an exact integer.

    The certificate holds when the distance to the nearest integer plus
    the provable float64 round-off bound stays strictly under 0.5 (and
    the value is finite); at exactly 0.5 the half-integer boundary is
    reachable, so such a lag is conservatively rejected.
    """
    return np.isfinite(values) & (
        np.abs(values - np.rint(values)) + bound < 0.5
    )


def _exact_cross_sums(
    ref_int: np.ndarray, cand_int: np.ndarray, lags: np.ndarray
) -> np.ndarray:
    """Exact pooled integer cross-sums at every requested lag, batched.

    Returns the exact ``sum ref[i]*cand[i+d]`` over the overlap of every
    channel for each lag d. Both int16 signals are split with
    ``x = hi*B + lo`` (B = SPLIT_BASE, floor decomposition, so
    0 <= lo < B also for negative x), giving the exact identity

        cross(d) = B**2 * C_hh(d) + B * (C_hl(d) + C_lh(d)) + C_ll(d)

    where each C(d) is the pooled per-channel cross-sum of the named
    parts, evaluated for *all* lags at once as a linear convolution with
    the reversed candidate, ``t = n_cand - 1 - d``. The four convolutions
    share two FFTs of the reference parts and two of the candidate parts;
    stereo channels are packed into the real/imaginary parts of one
    complex signal, so the real part of the product pools the two
    same-channel convolutions (the imaginary cross-channel terms are
    discarded).

    Each float64 convolution is certified to round to the exact integer.
    The standard radix-2 FFT convolution round-off bound (Schatzman, 1996:
    |error| <= 8 * eps * log2(N) * ||p||_2 * ||q||_2 for the full padded
    inputs, eps = 2**-53) is evaluated per component from its total
    signal energy. Because both sides are split into parts of magnitude
    at most B, every component bound stays below ~2.3e-3 at the maximum
    2**20-point FFT even for worst-case int16 inputs -- more than two
    hundred times under the 0.5 rounding distance. A lag that
    nevertheless cannot be certified (NaN/Inf or a bound reaching 0.5) is
    recomputed with a direct exact integer dot product, so the returned
    cross-sum is exact by construction.
    """
    n_ref, nch = ref_int.shape
    n_cand = cand_int.shape[0]
    fft_len = 1 << (n_ref + n_cand - 2).bit_length()
    assert fft_len.bit_length() - 1 <= _MAX_FFT_LOG2

    r_hi = ref_int // SPLIT_BASE
    r_lo = ref_int - r_hi * SPLIT_BASE
    c_hi = cand_int // SPLIT_BASE
    c_lo = cand_int - c_hi * SPLIT_BASE

    def packed(
        first_part: np.ndarray,
        second_part: np.ndarray | None,
        negate_second: bool = False,
    ) -> np.ndarray:
        """Channel 0 in the real part, channel 1 in +/- the imaginary part.

        Reference parts are packed with a minus sign (a - i b) and
        candidate parts with a plus (c + i d), so the real part of the
        product is conv(a,c) + conv(b,d): the two same-channel
        convolutions pooled, with no cross-channel term.
        """
        z = np.zeros(fft_len, dtype=np.complex128)
        z.real = _pad_signal(first_part[:, 0], fft_len, first_part.shape[0])
        if second_part is not None:
            col = _pad_signal(
                second_part[:, 1], fft_len, second_part.shape[0]
            )
            z.imag = -col if negate_second else col
        return z

    # Four complex forward FFTs and four inverse FFTs cover the four
    # component pairs (see the identity in the docstring). Mono inputs
    # leave the imaginary channel empty.
    r_hi2 = r_hi if nch == 2 else None
    r_lo2 = r_lo if nch == 2 else None
    c_hi_rev = c_hi[::-1]
    c_lo_rev = c_lo[::-1]
    f_rh = np.fft.fft(packed(r_hi, r_hi2, negate_second=True))
    f_rl = np.fft.fft(packed(r_lo, r_lo2, negate_second=True))
    f_ch = np.fft.fft(
        packed(c_hi_rev, c_hi_rev if nch == 2 else None)
    )
    f_cl = np.fft.fft(
        packed(c_lo_rev, c_lo_rev if nch == 2 else None)
    )
    hh = np.fft.ifft(f_rh * f_ch).real
    ll = np.fft.ifft(f_rl * f_cl).real
    hl = np.fft.ifft(f_rh * f_cl).real
    lh = np.fft.ifft(f_rl * f_ch).real
    del f_rh, f_rl, f_ch, f_cl

    rows = n_cand - 1 - lags
    c_hh = hh[rows]
    c_ll = ll[rows]
    c_hl = hl[rows]
    c_lh = lh[rows]
    del hh, ll, hl, lh

    # Full-signal L2 energies give the unconditional per-component
    # round-off certificate (no per-lag segment assumptions). The factor
    # uses _CERT_FFT_LOG2, an upper bound on the actual transform length,
    # so the certificate is valid uniformly for every compliant input
    # without recomputing it per transform size.
    e_rh = float(np.sum(r_hi * r_hi))
    e_rl = float(np.sum(r_lo * r_lo))
    e_ch = float(np.sum(c_hi * c_hi))
    e_cl = float(np.sum(c_lo * c_lo))
    roundoff = 8.0 * np.finfo(np.float64).eps * _CERT_FFT_LOG2
    bounds = (
        roundoff * math.sqrt(e_rh * e_ch),
        roundoff * math.sqrt(e_rh * e_cl),
        roundoff * math.sqrt(e_rl * e_ch),
        roundoff * math.sqrt(e_rl * e_cl),
    )

    rounded = (np.rint(c_hh), np.rint(c_hl), np.rint(c_lh), np.rint(c_ll))
    raw = (c_hh, c_hl, c_lh, c_ll)
    certified = np.ones(lags.shape[0], dtype=bool)
    for values, bound in zip(raw, bounds):
        certified &= _roundoff_certified(values, bound)
    cross = (
        rounded[0].astype(np.int64) * SPLIT_BASE * SPLIT_BASE
        + (rounded[1].astype(np.int64) + rounded[2].astype(np.int64)) * SPLIT_BASE
        + rounded[3].astype(np.int64)
    )
    if not bool(np.all(certified)):
        # Provably safe fallback: recompute only uncertified lags exactly.
        for lag in lags[~certified]:
            exact = _exact_overlap_stats(ref_int, cand_int, int(lag))
            cross[int(lag) - int(lags[0])] = exact[0]
    return cross


def _select_offset(
    ref: np.ndarray, cand: np.ndarray
) -> tuple[int, float, float]:
    """Pick the best shift.

    Returns (lag, cross_sum, ref_energy) for the winning lag. Ties on
    |correlation| go to the smallest |lag|, then the numerically smallest
    lag. All comparisons use exact integer statistics; the batched FFT
    supplies every lag's cross-sum in O(N log N) instead of one
    full-length dot product per near-tied peak.
    """
    n_ref, n_cand = ref.shape[0], cand.shape[0]
    # cand frame i+d aligns with ref frame i: d ranges in [-(n_ref-1), n_cand-1].
    lag_min = max(-MAX_SHIFT_SAMPLES, -(n_ref - 1))
    lag_max = min(MAX_SHIFT_SAMPLES, n_cand - 1)
    if lag_min > lag_max:
        raise AudioValidationError("Files are too short to overlap within +/-100 ms")
    lags = np.arange(lag_min, lag_max + 1, dtype=np.int64)

    # Decoded samples are exact int16/32768, so rint recovers the PCM
    # integers exactly; every statistic below is integer arithmetic.
    scale = float(1 << 15)
    ref_int = np.rint(ref * scale).astype(np.int64)
    cand_int = np.rint(cand * scale).astype(np.int64)

    cross_sums = _exact_cross_sums(ref_int, cand_int, lags)

    # Exact per-lag overlap energies pooled over channels.
    cs_ref = _int_cumsum_energy(ref_int)
    cs_cand = _int_cumsum_energy(cand_int)
    r_lo = np.maximum(-lags, 0)
    c_lo = np.maximum(lags, 0)
    overlap = np.minimum(n_ref - r_lo, n_cand - c_lo)
    e_ref = cs_ref[r_lo + overlap] - cs_ref[r_lo]
    e_cand = cs_cand[c_lo + overlap] - cs_cand[c_lo]

    nonzero = (e_ref > 0) & (e_cand > 0)
    if not bool(np.any(nonzero)):
        raise AudioValidationError(
            "Overlapping region is silent; cannot estimate gain or offset"
        )

    # Lags are visited in tie-break order (ascending |lag|, then lag);
    # the incumbent can only be displaced by a strictly larger exact
    # |correlation|, so equal lags later in the order never replace it.
    # Once the incumbent reaches exactly |corr| = 1 the scan stops --
    # nothing can exceed it.
    order = np.lexsort((lags, np.abs(lags)))
    best_idx = -1
    best_stats: tuple[int, int, int] | None = None
    for idx in order.tolist():
        if not bool(nonzero[idx]):
            continue
        if best_stats is not None:
            x_b, er_b, ec_b = best_stats
            if x_b * x_b == er_b * ec_b:
                break
        stats = (
            int(cross_sums[idx]),
            int(e_ref[idx]),
            int(e_cand[idx]),
        )
        if best_stats is None or _corr_exceeds(stats, best_stats):
            best_stats = stats
            best_idx = int(idx)

    # lag 0 is always in range and nonzero inputs give it positive energy,
    # so the loop always sets an incumbent.
    assert best_stats is not None and best_idx >= 0
    cross, e_ref_win, _ = best_stats
    return int(lags[best_idx]), cross / scale**2, e_ref_win / scale**2


def _anomaly_intervals(
    ref: np.ndarray,
    cand: np.ndarray,
    lag: int,
    gain: float,
    ref_rms: float,
) -> list[tuple[int, int]]:
    """Flag 100 ms reference-timeline windows whose residual RMS is too high.

    Consecutive flagged windows are merged into half-open sample intervals.
    Windows with no overlap (beyond the aligned region) are not scored.
    """
    n_ref = ref.shape[0]
    r_lo, _, overlap = _overlap_slice((n_ref, cand.shape[0]), lag)
    r_hi = r_lo + overlap
    threshold = max(RESIDUAL_FLOOR, RESIDUAL_RELATIVE * ref_rms)

    flagged: list[int] = []
    starts = np.arange(0, n_ref, WINDOW_SAMPLES)
    for k, w_lo in enumerate(starts):
        w_hi = min(int(w_lo) + WINDOW_SAMPLES, n_ref)
        lo = max(int(w_lo), r_lo)
        hi = min(w_hi, r_hi)
        if hi <= lo:
            continue
        ref_seg = ref[lo:hi]
        cand_seg = cand[lo + lag : hi + lag]
        residual = cand_seg - gain * ref_seg
        rms = float(np.sqrt(np.mean(residual * residual)))
        if rms > threshold:
            flagged.append(k)

    intervals: list[tuple[int, int]] = []
    if not flagged:
        return intervals
    run_start = prev = flagged[0]
    for k in flagged[1:] + [None]:
        if k is not None and k == prev + 1:
            prev = k
            continue
        start = int(run_start) * WINDOW_SAMPLES
        end = min((int(prev) + 1) * WINDOW_SAMPLES, n_ref)
        intervals.append((start, end))
        if k is not None:
            run_start = prev = k
    return intervals


def compare_wav_gain(reference_wav: bytes, candidate_wav: bytes) -> CompareGainResult:
    ref = _read_wav(reference_wav)
    cand = _read_wav(candidate_wav)

    if ref.shape[1] != cand.shape[1]:
        raise AudioValidationError(
            f"Channel count mismatch: reference has {ref.shape[1]}, "
            f"candidate has {cand.shape[1]}"
        )
    if not np.any(ref) or not np.any(cand):
        raise AudioValidationError("Input audio must not be completely silent")

    lag, cross, e_ref = _select_offset(ref, cand)
    if e_ref <= 0.0 or not np.isfinite(cross):
        raise AudioValidationError(
            "Overlapping region is silent; cannot estimate gain or offset"
        )

    # One signed gain shared by every channel: least squares over the whole
    # aligned overlap of all channels.
    gain = cross / e_ref
    if not np.isfinite(gain) or gain == 0.0:
        raise AudioValidationError("Cannot estimate a finite gain from the overlap")

    r_lo, _, overlap = _overlap_slice((ref.shape[0], cand.shape[0]), lag)
    ref_overlap = ref[r_lo : r_lo + overlap]
    ref_rms = float(np.sqrt(np.mean(ref_overlap * ref_overlap)))

    intervals = _anomaly_intervals(ref, cand, lag, gain, ref_rms)

    return CompareGainResult(
        offset_samples=lag,
        gain_db=float(20.0 * np.log10(abs(gain))),
        polarity_inverted=bool(gain < 0.0),
        reference_unmatched_samples=ref.shape[0] - overlap,
        candidate_unmatched_samples=cand.shape[0] - overlap,
        anomaly_intervals=intervals,
        overlap_gain_only=not intervals,
    )
