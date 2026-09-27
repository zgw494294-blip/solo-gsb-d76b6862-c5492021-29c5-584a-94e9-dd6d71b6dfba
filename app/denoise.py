"""Spectral-subtraction denoising driven by a user-marked noise segment.

``noise_interval`` is a half-open [start, end) interval of input sample
indices marking a noise-only reference region. Per channel the signal is
analysed with a 1024-point Hann window at a 256-sample hop; the per-bin
median power over the frames fully inside the reference region is the
noise estimate N. Every frame of the whole clip is then scaled by
G = max(0.1, sqrt(max(0, 1 - strength * N / (|X|^2 + 1e-12)))), the gain
is smoothed along time with a 3-frame moving average, and the signal is
rebuilt by overlap-add. The output keeps the input frame count, channel
count and 16-bit format; strength = 0 leaves every PCM sample unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .loudness import (
    SAMPLE_RATE,
    AudioValidationError,
    _quantize,
    _read_wav,
    _write_wav,
)

N_FFT = 1024
HOP = 256
# Left padding (samples) so every input sample sits under exactly
# N_FFT // HOP windows and the squared-window overlap sum is constant.
LEAD_PAD = N_FFT - HOP
MIN_NOISE_FRAMES = SAMPLE_RATE // 5   # reference length: 0.2-2 s
MAX_NOISE_FRAMES = 2 * SAMPLE_RATE
MIN_GAIN = 0.1
EPSILON = 1e-12

# Periodic Hann: COLA-compliant at a N_FFT/4 hop (squared overlap sum 1.5).
WINDOW = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT)


@dataclass
class DenoiseResult:
    wav_bytes: bytes


def _validate_noise_interval(
    noise_interval: tuple[int, int], n_frames: int
) -> tuple[int, int]:
    """Check the half-open reference interval and return it as plain ints.

    The interval must be non-empty, inside [0, n_frames) and 0.2-2 s long.
    """
    start, end = int(noise_interval[0]), int(noise_interval[1])
    if not 0 <= start < end <= n_frames:
        raise AudioValidationError(
            f"noise_interval [{start}, {end}) is empty or outside "
            f"[0, {n_frames})"
        )
    length = end - start
    if not MIN_NOISE_FRAMES <= length <= MAX_NOISE_FRAMES:
        raise AudioValidationError(
            f"noise_interval [{start}, {end}) is {length} frames long; the "
            f"reference must be {MIN_NOISE_FRAMES}-{MAX_NOISE_FRAMES} frames "
            "(0.2-2 s)"
        )
    return start, end


def _stft(channel: np.ndarray) -> np.ndarray:
    """RFFT spectra (n_frames, N_FFT // 2 + 1) of the zero-padded signal."""
    n = channel.shape[0]
    # Frame k starts at k*HOP - LEAD_PAD in input coordinates; the last
    # frame starts no later than n - 1 so the tail is fully covered.
    n_frames = (n - 1 + LEAD_PAD) // HOP + 1
    padded = np.zeros(LEAD_PAD + n_frames * HOP, dtype=np.float64)
    padded[LEAD_PAD : LEAD_PAD + n] = channel
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP]
    return np.fft.rfft(frames * WINDOW, axis=1)


def _istft(spectra: np.ndarray, n: int) -> np.ndarray:
    """Overlap-add the windowed frames and normalize by the window power."""
    frames = np.fft.irfft(spectra, n=N_FFT, axis=1) * WINDOW
    padded = np.zeros(LEAD_PAD + spectra.shape[0] * HOP, dtype=np.float64)
    window_power = np.zeros_like(padded)
    for k in range(spectra.shape[0]):
        offset = k * HOP
        padded[offset : offset + N_FFT] += frames[k]
        window_power[offset : offset + N_FFT] += WINDOW * WINDOW
    return padded[LEAD_PAD : LEAD_PAD + n] / window_power[LEAD_PAD : LEAD_PAD + n]


def _smooth_gain(gain: np.ndarray) -> np.ndarray:
    """3-frame moving average along the time axis (edges replicated)."""
    padded = np.pad(gain, ((1, 1), (0, 0)), mode="edge")
    return (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0


def _denoise_channel(
    channel: np.ndarray, start: int, end: int, strength: float
) -> np.ndarray:
    n = channel.shape[0]
    spectra = _stft(channel)
    power = np.abs(spectra) ** 2
    # Frames fully inside the reference interval (padded coordinates).
    first = (start + LEAD_PAD + HOP - 1) // HOP
    last = (end + LEAD_PAD - N_FFT) // HOP
    noise_power = np.median(power[first : last + 1], axis=0)
    gain = np.sqrt(
        np.maximum(0.0, 1.0 - strength * noise_power / (power + EPSILON))
    )
    gain = np.maximum(MIN_GAIN, gain)
    gain = _smooth_gain(gain)
    return _istft(spectra * gain, n)


def denoise_wav(
    wav_bytes: bytes, noise_interval: tuple[int, int], strength: float
) -> DenoiseResult:
    samples = _read_wav(wav_bytes)
    n_frames, channels = samples.shape

    start, end = _validate_noise_interval(noise_interval, n_frames)
    for ch in range(channels):
        if not np.any(samples[start:end, ch]):
            raise AudioValidationError(
                f"noise_interval [{start}, {end}) is all-zero on channel "
                f"{ch}; the reference must contain noise samples on every "
                "channel"
            )

    denoised = np.empty_like(samples)
    for ch in range(channels):
        denoised[:, ch] = _denoise_channel(samples[:, ch], start, end, strength)

    pcm = _quantize(denoised)
    return DenoiseResult(wav_bytes=_write_wav(pcm, channels))
