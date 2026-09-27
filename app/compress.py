"""Feed-forward dynamic-range compressor with a channel-linked envelope.

Per frame the detector level p is the maximum absolute sample value across
channels. The envelope e (initial value 0) follows

    e = a * e_prev + (1 - a) * p,   a = exp(-1 / (48 * m))

where m is attack_ms while p exceeds the previous envelope value and
release_ms otherwise (48 kHz => 48 samples per millisecond). With
L = 20*log10(max(e, 1e-12)), T = threshold_dbfs, R = ratio and K = knee_db
the target level is

    L                                   for L <= T - K/2
    T + (L - T) / R                     for L >= T + K/2
    L + (1/R - 1) * (L - T + K/2)^2 / (2K)   in between (soft knee)

K = 0 selects the hard knee, i.e. only the first two branches. The
difference between the target and L is applied as one shared gain to every
channel, so the stereo image is preserved. The output keeps the input
frame count, channel count and 16-bit format; ratio = 1 leaves every PCM
sample unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .loudness import SAMPLE_RATE, _quantize, _read_wav, _write_wav

# Envelope floor for the dB conversion, keeps log10 finite on silence.
ENVELOPE_FLOOR = 1e-12
# Samples per millisecond at the fixed 48 kHz sample rate.
SAMPLES_PER_MS = SAMPLE_RATE // 1000  # 48


@dataclass
class CompressResult:
    wav_bytes: bytes


def _envelope(peak: np.ndarray, attack_ms: float, release_ms: float) -> np.ndarray:
    """Per-frame detector envelope of the channel-max peak signal.

    Smoothing coefficient a = exp(-1 / (48 * m)) with m = attack_ms on
    rising frames (p above the previous envelope) and release_ms otherwise.
    """
    attack_a = math.exp(-1.0 / (SAMPLES_PER_MS * attack_ms))
    release_a = math.exp(-1.0 / (SAMPLES_PER_MS * release_ms))
    env = np.empty(peak.shape[0], dtype=np.float64)
    e = 0.0
    for i, p in enumerate(peak.tolist()):
        a = attack_a if p > e else release_a
        e = a * e + (1.0 - a) * p
        env[i] = e
    return env


def _target_level(
    level: np.ndarray, threshold_dbfs: float, ratio: float, knee_db: float
) -> np.ndarray:
    """Static compression curve: target level (dB) per envelope level (dB)."""
    half_knee = knee_db / 2.0
    target = level.copy()
    over = level >= threshold_dbfs + half_knee
    if knee_db > 0.0:
        in_knee = (level > threshold_dbfs - half_knee) & ~over
        x = level[in_knee] - threshold_dbfs + half_knee
        target[in_knee] = level[in_knee] + (1.0 / ratio - 1.0) * (
            x * x
        ) / (2.0 * knee_db)
    target[over] = threshold_dbfs + (level[over] - threshold_dbfs) / ratio
    return target


def compress_wav(
    wav_bytes: bytes,
    threshold_dbfs: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    knee_db: float,
) -> CompressResult:
    samples = _read_wav(wav_bytes)
    n_frames, channels = samples.shape

    peak = np.max(np.abs(samples), axis=1)
    env = _envelope(peak, attack_ms, release_ms)
    level = 20.0 * np.log10(np.maximum(env, ENVELOPE_FLOOR))

    target = _target_level(level, threshold_dbfs, ratio, knee_db)
    gain = np.power(10.0, (target - level) / 20.0)

    pcm = _quantize(samples * gain[:, np.newaxis])
    return CompressResult(wav_bytes=_write_wav(pcm, channels))
