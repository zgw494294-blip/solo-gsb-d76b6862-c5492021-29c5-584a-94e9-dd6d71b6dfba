"""Spectral-flux onset detection for edit-point localisation.

Each channel is analysed with a centered 1024-point Hann window at a
256-sample hop (real FFT, signal zero-padded at both ends so frame k is
centered on sample k*256). Per frame the spectral flux is the sum of the
positive per-bin magnitude differences against the previous frame (the
signal is zero before the first frame, so the first frame is compared
against an all-zero spectrum); for stereo the per-frame flux is the
larger of the two channels. A frame becomes an onset candidate when it
is a strict local peak (a plateau peak counts at its earliest frame),
its flux is strictly above the median + 3*MAD of the 20 frames on either
side, and it reaches ``min_relative_flux`` times the whole-file maximum
flux. Candidates are then selected strongest first (ties: earliest),
keeping only those at least 80 ms away from every already selected
onset. The audio itself is never modified.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .loudness import _read_wav

N_FFT = 1024
HOP = 256
PAD = N_FFT // 2  # zero padding at both ends: frame k is centered on k*HOP
NEIGHBOURHOOD = 20  # frames on each side for the adaptive threshold
MAD_SCALE = 3.0
MIN_SEPARATION_HOPS = 15  # 80 ms = 3840 samples = 15 hops

# Periodic Hann window (same convention as the denoise STFT).
WINDOW = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT)


@dataclass
class Onset:
    frame_index: int  # window-centre sample index, clamped to [0, n-1]
    strength: float  # spectral flux relative to the whole-file maximum


@dataclass
class OnsetsResult:
    onsets: list[Onset]  # ascending time order


def _spectral_flux(channel: np.ndarray) -> np.ndarray:
    """Per-frame spectral flux of one channel.

    Frame k is centered on sample k*HOP (the signal is zero-padded by PAD
    samples on both ends); its flux is the sum of the positive per-bin
    magnitude differences against the previous frame, with an all-zero
    spectrum before the first frame.
    """
    n = channel.shape[0]
    padded = np.zeros(n + 2 * PAD, dtype=np.float64)
    padded[PAD : PAD + n] = channel
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP]
    magnitudes = np.abs(np.fft.rfft(frames * WINDOW, axis=1))
    increases = np.maximum(0.0, np.diff(magnitudes, axis=0, prepend=0.0))
    return increases.sum(axis=1)


def _is_local_peak(flux: np.ndarray, i: int) -> bool:
    """Strict local maximum; a plateau peak counts at its earliest frame."""
    if i > 0 and flux[i] <= flux[i - 1]:
        return False  # not strictly rising from the left
    j = i
    while j + 1 < flux.shape[0] and flux[j + 1] == flux[i]:
        j += 1  # end of the plateau starting at i
    return j + 1 == flux.shape[0] or flux[j + 1] < flux[i]


def _passes_adaptive_threshold(flux: np.ndarray, i: int) -> bool:
    """flux[i] strictly above median + 3*MAD of the 20 frames on each side."""
    lo = max(0, i - NEIGHBOURHOOD)
    hi = min(flux.shape[0], i + NEIGHBOURHOOD + 1)
    neighbours = np.concatenate((flux[lo:i], flux[i + 1 : hi]))
    median = float(np.median(neighbours))
    mad = float(np.median(np.abs(neighbours - median)))
    return flux[i] > median + MAD_SCALE * mad


def detect_onsets(wav_bytes: bytes, min_relative_flux: float) -> OnsetsResult:
    samples = _read_wav(wav_bytes)
    n, channels = samples.shape

    flux = _spectral_flux(samples[:, 0])
    for ch in range(1, channels):
        flux = np.maximum(flux, _spectral_flux(samples[:, ch]))

    max_flux = float(flux.max())
    if max_flux <= 0.0:
        return OnsetsResult(onsets=[])  # all silence: no transients

    threshold = min_relative_flux * max_flux
    candidates = [
        (i, float(flux[i]))
        for i in range(flux.shape[0])
        if flux[i] >= threshold
        and _is_local_peak(flux, i)
        and _passes_adaptive_threshold(flux, i)
    ]

    # Strongest first (ties: earliest); keep onsets >= 80 ms from all kept.
    candidates.sort(key=lambda candidate: (-candidate[1], candidate[0]))
    selected: list[tuple[int, float]] = []
    for i, value in candidates:
        if all(abs(i - j) >= MIN_SEPARATION_HOPS for j, _ in selected):
            selected.append((i, value))

    selected.sort(key=lambda candidate: candidate[0])  # ascending time
    return OnsetsResult(
        onsets=[
            Onset(
                frame_index=min(i * HOP, n - 1),
                strength=round(value / max_flux, 6),
            )
            for i, value in selected
        ]
    )
