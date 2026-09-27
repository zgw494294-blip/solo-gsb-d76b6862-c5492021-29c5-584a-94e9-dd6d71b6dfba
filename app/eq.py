"""Parametric EQ: cascaded RBJ peaking biquads under a true-peak ceiling.

Each requested band is a peaking-EQ biquad designed with the RBJ Audio
EQ Cookbook formulas at the fixed 48 kHz sample rate:

    A = 10^(gain_db / 40)
    w0 = 2 * pi * frequency_hz / Fs,  alpha = sin(w0) / (2 * Q)
    b = [1 + alpha*A, -2*cos(w0), 1 - alpha*A]
    a = [1 + alpha/A, -2*cos(w0), 1 - alpha/A]

normalized so a0 = 1. The bands are cascaded in request order, every
band applied with zero initial state; channels are filtered
independently but share the same per-band coefficients. After filtering
a single non-positive gain is applied to the whole clip so the
4x-oversampled true peak stays under ceiling_dbtp; the quantized output
is re-measured and the gain backed off further if the ceiling is still
exceeded. No clipping: the output keeps the input frame count, channel
count and 16-bit format.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from .loudness import (
    MAX_GAIN_BACKOFF_STEPS,
    SAMPLE_RATE,
    _quantize,
    _read_wav,
    _write_wav,
    true_peak_dbtp,
)


@dataclass
class EqBand:
    frequency_hz: float
    q: float
    gain_db: float


@dataclass
class EqResult:
    wav_bytes: bytes
    input_true_peak_dbtp: float
    output_true_peak_dbtp: float
    post_gain_db: float


def peaking_coefficients(
    frequency_hz: float, q: float, gain_db: float
) -> tuple[np.ndarray, np.ndarray]:
    """RBJ Audio EQ Cookbook peaking-EQ biquad at 48 kHz, a0 normalized to 1."""
    a_gain = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * frequency_hz / SAMPLE_RATE
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)
    a0 = 1.0 + alpha / a_gain
    b = np.array([1.0 + alpha * a_gain, -2.0 * cos_w0, 1.0 - alpha * a_gain]) / a0
    a = np.array([a0, -2.0 * cos_w0, 1.0 - alpha / a_gain]) / a0
    return b, a


def eq_wav(wav_bytes: bytes, bands: list[EqBand], ceiling_dbtp: float) -> EqResult:
    samples = _read_wav(wav_bytes)
    _, channels = samples.shape

    input_peak = true_peak_dbtp(samples)

    filtered = samples
    for band in bands:
        b, a = peaking_coefficients(band.frequency_hz, band.q, band.gain_db)
        # Zero initial state; each channel (column) filtered independently.
        filtered = lfilter(b, a, filtered, axis=0)

    # One non-positive gain for the whole clip: only what the true-peak
    # ceiling requires, never a boost.
    post_gain_db = min(0.0, ceiling_dbtp - true_peak_dbtp(filtered))

    pcm_out = np.empty(0, dtype="<i2")
    out_peak = float("-inf")
    for _ in range(MAX_GAIN_BACKOFF_STEPS):
        pcm_out = _quantize(filtered * 10.0 ** (post_gain_db / 20.0))
        # Re-measure the quantized output exactly as it will be delivered.
        restored = pcm_out.astype(np.float64) / 32768.0
        out_peak = true_peak_dbtp(restored)
        if out_peak <= ceiling_dbtp + 1e-6:
            break
        # Quantization/oversampling pushed the peak over: back the gain off.
        post_gain_db -= (out_peak - ceiling_dbtp) + 1e-3

    return EqResult(
        wav_bytes=_write_wav(pcm_out, channels),
        input_true_peak_dbtp=input_peak,
        output_true_peak_dbtp=out_peak,
        post_gain_db=post_gain_db,
    )
