"""Core loudness normalization logic (EBU R128 / BS.1770).

Pipeline: decode & validate WAV -> measure integrated loudness (pyloudnorm)
and true peak (4x polyphase resampling) -> pick a single static gain of
min(gain-to-target, gain-to-ceiling) -> apply, quantize back to 16-bit PCM,
re-measure the quantized signal and back the gain off if the ceiling is
exceeded. No compression, no clipping.
"""
from __future__ import annotations

import io
import wave
from dataclasses import dataclass

import numpy as np
import pyloudnorm as pyln
from scipy.signal import resample_poly

SAMPLE_RATE = 48_000
MIN_DURATION_S = 1.0
MAX_DURATION_S = 10.0
TRUE_PEAK_OVERSAMPLING = 4
TARGET_TOLERANCE_LU = 0.1
MAX_GAIN_BACKOFF_STEPS = 10
# Generous upper bound: 10 s of 48 kHz / 16-bit / stereo PCM is ~1.9 MB.
MAX_WAV_BYTES = 16 * 1024 * 1024

# BS.1770 K-weighted integrated loudness meter (filter coefficients are
# stateless per measurement, so one shared meter is fine).
_METER = pyln.Meter(SAMPLE_RATE)


class AudioValidationError(ValueError):
    """Input audio does not meet the API requirements (maps to HTTP 422)."""


@dataclass
class NormalizeResult:
    wav_bytes: bytes
    input_loudness_lufs: float
    output_loudness_lufs: float
    input_true_peak_dbtp: float
    output_true_peak_dbtp: float
    gain_db: float
    target_reached: bool


def _read_wav(
    wav_bytes: bytes,
    min_duration_s: float = MIN_DURATION_S,
    max_duration_s: float = MAX_DURATION_S,
) -> np.ndarray:
    """Decode a 48 kHz / 16-bit PCM / mono|stereo WAV.

    Returns float64 samples in [-1, 1] with shape (n_frames, n_channels).
    """
    if len(wav_bytes) > MAX_WAV_BYTES:
        raise AudioValidationError("WAV payload too large")
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            frames = wf.getnframes()
            if wf.getcomptype() != "NONE":
                raise AudioValidationError("WAV must be uncompressed PCM")
            if width != 2:
                raise AudioValidationError(
                    f"WAV must be 16-bit PCM (got {width * 8}-bit)"
                )
            if rate != SAMPLE_RATE:
                raise AudioValidationError(f"WAV must be 48 kHz (got {rate} Hz)")
            if channels not in (1, 2):
                raise AudioValidationError(
                    f"WAV must be mono or stereo (got {channels} channels)"
                )
            duration = frames / rate
            if not min_duration_s <= duration <= max_duration_s:
                raise AudioValidationError(
                    f"WAV duration must be {min_duration_s:g}-"
                    f"{max_duration_s:g} s (got {duration:.3f} s)"
                )
            raw = wf.readframes(frames)
    except AudioValidationError:
        raise
    except (wave.Error, EOFError) as exc:
        raise AudioValidationError(f"Cannot parse WAV file: {exc}") from exc

    pcm = np.frombuffer(raw, dtype="<i2")
    if pcm.size != frames * channels:
        raise AudioValidationError("Corrupt WAV: data chunk size mismatch")
    return pcm.astype(np.float64).reshape(-1, channels) / 32768.0


def _write_wav(pcm: np.ndarray, channels: int) -> bytes:
    """Encode int16 PCM (n_frames, n_channels) as a 48 kHz WAV."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def integrated_loudness(samples: np.ndarray) -> float:
    """EBU R128 / BS.1770 integrated loudness in LUFS."""
    return float(_METER.integrated_loudness(samples))


def true_peak_dbtp(samples: np.ndarray) -> float:
    """Highest true peak in dBTP, estimated on the 4x oversampled signal."""
    oversampled = resample_poly(samples, TRUE_PEAK_OVERSAMPLING, 1, axis=0)
    peak = float(np.max(np.abs(oversampled))) if oversampled.size else 0.0
    if peak <= 0.0:
        return float("-inf")
    return 20.0 * np.log10(peak)


def _quantize(samples: np.ndarray) -> np.ndarray:
    """Float [-1, 1] -> 16-bit PCM, shape preserved."""
    scaled = np.clip(samples, -1.0, 1.0) * 32768.0
    return np.clip(np.round(scaled), -32768, 32767).astype("<i2")


def normalize_wav(
    wav_bytes: bytes, target_lufs: float, ceiling_dbtp: float
) -> NormalizeResult:
    samples = _read_wav(wav_bytes)
    channels = samples.shape[1]

    input_loudness = integrated_loudness(samples)
    if not np.isfinite(input_loudness):
        raise AudioValidationError(
            "Audio is silent or below the loudness measurement gate"
        )
    input_peak = true_peak_dbtp(samples)

    # One static gain for the whole file: the smaller of what the loudness
    # target needs and what the true-peak ceiling allows.
    gain_db = min(target_lufs - input_loudness, ceiling_dbtp - input_peak)

    pcm_out = np.empty(0, dtype="<i2")
    out_loudness = out_peak = float("-inf")
    for _ in range(MAX_GAIN_BACKOFF_STEPS):
        pcm_out = _quantize(samples * 10.0 ** (gain_db / 20.0))
        # Re-measure the quantized output exactly as it will be delivered.
        restored = pcm_out.astype(np.float64) / 32768.0
        out_loudness = integrated_loudness(restored)
        out_peak = true_peak_dbtp(restored)
        if out_peak <= ceiling_dbtp + 1e-6:
            break
        # Quantization/oversampling pushed the peak over: back the gain off.
        gain_db -= (out_peak - ceiling_dbtp) + 1e-3

    return NormalizeResult(
        wav_bytes=_write_wav(pcm_out, channels),
        input_loudness_lufs=input_loudness,
        output_loudness_lufs=out_loudness,
        input_true_peak_dbtp=input_peak,
        output_true_peak_dbtp=out_peak,
        gain_db=gain_db,
        target_reached=abs(out_loudness - target_lufs) <= TARGET_TOLERANCE_LU,
    )
