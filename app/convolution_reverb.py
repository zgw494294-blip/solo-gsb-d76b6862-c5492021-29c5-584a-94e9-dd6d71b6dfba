"""Convolution reverb: FFT linear convolution with a user impulse response.

The source WAV (48 kHz / 16-bit PCM / mono|stereo / 1-10 s) is convolved
with an impulse-response WAV (same format rules, but 1-2 s long and not
all silent) using FFT-based linear convolution, so the reverb tail
extends the output to N + M - 1 frames. A mono IR is applied to every
source channel; a stereo IR with a mono source duplicates the dry signal
across both output channels, and with a stereo source convolves
channel-by-channel. Each output channel mixes

    (1 - wet) * zero-padded dry + wet * convolved wet

When the mix exceeds the range 16-bit PCM can represent, one shared
non-positive gain is applied to all channels before quantization (the
signal is never clipped); the actually applied gain is reported as
post_gain_db (0 when no attenuation was needed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from .loudness import (
    AudioValidationError,
    _quantize,
    _read_wav,
    _write_wav,
)

IR_MIN_DURATION_S = 1.0
IR_MAX_DURATION_S = 2.0
# Largest magnitudes 16-bit PCM represents without quantization clipping:
# [-1.0, 32767/32768].
PCM_POSITIVE_FULL_SCALE = 32767.0 / 32768.0
PCM_NEGATIVE_FULL_SCALE = 1.0


@dataclass
class ConvolutionReverbResult:
    wav_bytes: bytes
    output_frames: int
    post_gain_db: float


def _read_ir_wav(ir_wav_bytes: bytes) -> np.ndarray:
    """Decode the impulse response: same format rules as the source, but
    1-2 s long and not all silent."""
    ir = _read_wav(
        ir_wav_bytes,
        min_duration_s=IR_MIN_DURATION_S,
        max_duration_s=IR_MAX_DURATION_S,
    )
    if not np.any(ir):
        raise AudioValidationError("Impulse response must not be all silence")
    return ir


def convolution_reverb_wav(
    wav_bytes: bytes, ir_wav_bytes: bytes, wet: float
) -> ConvolutionReverbResult:
    samples = _read_wav(wav_bytes)
    ir = _read_ir_wav(ir_wav_bytes)
    n, source_channels = samples.shape
    m, ir_channels = ir.shape

    # Mono IR feeds every source channel; stereo IR with a mono source
    # duplicates the dry signal; otherwise channel-by-channel.
    out_channels = max(source_channels, ir_channels)
    out_frames = n + m - 1
    mixed = np.zeros((out_frames, out_channels), dtype=np.float64)
    for ch in range(out_channels):
        dry = samples[:, ch] if source_channels == 2 else samples[:, 0]
        impulse = ir[:, ch] if ir_channels == 2 else ir[:, 0]
        # FFT linear convolution: full output is N + M - 1 frames.
        mixed[:, ch] = wet * fftconvolve(dry, impulse, mode="full")
        mixed[:n, ch] += (1.0 - wet) * dry  # dry zero-padded to the tail

    # One shared non-positive gain whenever the mix leaves the 16-bit PCM
    # range; the quantized output is never clipped.
    gain = 1.0
    pos_peak = float(mixed.max())
    neg_peak = float(-mixed.min())
    if pos_peak > PCM_POSITIVE_FULL_SCALE:
        gain = min(gain, PCM_POSITIVE_FULL_SCALE / pos_peak)
    if neg_peak > PCM_NEGATIVE_FULL_SCALE:
        gain = min(gain, PCM_NEGATIVE_FULL_SCALE / neg_peak)
    post_gain_db = 20.0 * math.log10(gain) if gain < 1.0 else 0.0

    pcm = _quantize(mixed * gain)
    return ConvolutionReverbResult(
        wav_bytes=_write_wav(pcm, out_channels),
        output_frames=out_frames,
        post_gain_db=post_gain_db,
    )
