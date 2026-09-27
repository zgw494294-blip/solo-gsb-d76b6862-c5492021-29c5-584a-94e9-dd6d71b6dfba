import base64
import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

RATE = 48000
N = 3 * RATE  # 3 s of audio (long enough for a >2 s interval to fit)
INTERVAL = [4800, 14400]  # 0.2 s calibration interval


def stereo_wav_b64(left: np.ndarray, right: np.ndarray, rate: int = RATE) -> str:
    pcm = np.stack([left, right], axis=1).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def mono_wav_b64(pcm: np.ndarray, rate: int = RATE) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.astype("<i2").tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def decode_wav(b64: str):
    raw = base64.b64decode(b64)
    with wave.open(io.BytesIO(raw), "rb") as wf:
        params = (wf.getnchannels(), wf.getsampwidth(), wf.getframerate())
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    return params, pcm


def make_source(n=N, seed=0, amp=8000):
    """Aperiodic noise, clipped clear of the full-scale edges."""
    rng = np.random.default_rng(seed)
    return np.clip(np.round(rng.standard_normal(n) * amp), -32000, 32000).astype(
        np.int64
    )


def delayed(src: np.ndarray, shift: int) -> np.ndarray:
    """out[j] = src[j - shift], zero-filled (shift > 0 delays the track)."""
    out = np.zeros_like(src)
    if shift >= 0:
        out[shift:] = src[: len(src) - shift]
    else:
        out[: len(src) + shift] = src[-shift:]
    return out


def post_align(left, right=None, interval=None, rate=RATE, mono=False):
    if interval is None:
        interval = INTERVAL
    if mono:
        payload = mono_wav_b64(left, rate)
    else:
        payload = stereo_wav_b64(left, right, rate)
    return client.post(
        "/v1/align-stereo",
        json={"wav_base64": payload, "calibration_interval": interval},
    )


def test_positive_offset_aligned():
    src = make_source(seed=0)
    shift = 37
    right = delayed(src, shift)
    resp = post_align(src, right)
    assert resp.status_code == 200
    body = resp.json()
    assert body["offset_samples"] == shift
    assert body["polarity_inverted"] is False
    assert body["correlation"] == pytest.approx(1.0)

    params, out = decode_wav(body["wav_base64"])
    assert params == (2, 2, RATE)
    out = out.reshape(-1, 2)
    assert out.shape == (N, 2)
    # Left track passed through sample-for-sample.
    assert np.array_equal(out[:, 0], src.astype("<i2"))
    # Right track realigned over its whole length: out[i] = right[i + shift].
    assert np.array_equal(out[: N - shift, 1], src[: N - shift].astype("<i2"))
    # Out-of-range tail frames are zero-filled.
    assert np.array_equal(out[N - shift :, 1], np.zeros(shift, dtype="<i2"))


def test_negative_offset_aligned():
    src = make_source(seed=1)
    shift = -120
    right = delayed(src, shift)
    resp = post_align(src, right)
    assert resp.status_code == 200
    body = resp.json()
    assert body["offset_samples"] == shift
    assert body["polarity_inverted"] is False

    _, out = decode_wav(body["wav_base64"])
    out = out.reshape(-1, 2)
    assert np.array_equal(out[:, 0], src.astype("<i2"))
    # Head frames fall out of range and are zero-filled.
    assert np.array_equal(out[: -shift, 1], np.zeros(-shift, dtype="<i2"))
    assert np.array_equal(out[-shift:, 1], src[-shift:].astype("<i2"))


@pytest.mark.parametrize("shift", [-240, 240])
def test_boundary_offsets_detected(shift):
    src = make_source(seed=2)
    resp = post_align(src, delayed(src, shift))
    assert resp.status_code == 200
    assert resp.json()["offset_samples"] == shift


def test_negative_correlation_inverts_right_track():
    src = make_source(seed=3)
    shift = 15
    right = -delayed(src, shift)  # delayed and polarity-inverted
    resp = post_align(src, right)
    assert resp.status_code == 200
    body = resp.json()
    assert body["offset_samples"] == shift
    assert body["polarity_inverted"] is True
    assert body["correlation"] == pytest.approx(-1.0)

    _, out = decode_wav(body["wav_base64"])
    out = out.reshape(-1, 2)
    # Inversion restores the source polarity on the aligned overlap.
    assert np.array_equal(out[: N - shift, 1], src[: N - shift].astype("<i2"))


def test_inversion_clamps_minus_full_scale():
    src = make_source(seed=4)
    shift = 20
    right = -delayed(src, shift)
    right[5000] = -32768  # plant the full-scale negative sample
    resp = post_align(src, right)
    assert resp.status_code == 200
    body = resp.json()
    assert body["polarity_inverted"] is True

    _, out = decode_wav(body["wav_base64"])
    out = out.reshape(-1, 2)
    # -(-32768) = 32768 must clamp to 32767, not wrap to -32768.
    assert out[5000 - shift, 1] == 32767


def test_zero_offset_positive_correlation_leaves_pcm_unchanged():
    src = make_source(seed=5)
    resp = post_align(src, src.copy())
    assert resp.status_code == 200
    body = resp.json()
    assert body["offset_samples"] == 0
    assert body["polarity_inverted"] is False
    assert body["correlation"] == pytest.approx(1.0)

    _, out = decode_wav(body["wav_base64"])
    assert np.array_equal(
        out.reshape(-1, 2), np.stack([src, src], axis=1).astype("<i2")
    )


def test_tie_picks_smallest_absolute_offset():
    # Period-2 pattern: every offset correlates at exactly +/-1, so the
    # tie-break collapses to offset 0.
    pattern = np.where(np.arange(N) % 2 == 0, 10000, -10000).astype(np.int64)
    resp = post_align(pattern, pattern.copy())
    assert resp.status_code == 200
    assert resp.json()["offset_samples"] == 0


def test_tie_picks_numerically_smallest_offset():
    # Period-200 source, right track delayed by exactly half a period:
    # offsets -100 and +100 both reach correlation exactly 1 and tie on
    # |offset|, so the numerically smallest wins.
    rng = np.random.default_rng(6)
    period = 200
    pattern = np.clip(
        np.round(rng.standard_normal(period) * 8000), -32000, 32000
    ).astype(np.int64)
    src = pattern[np.arange(N) % period]
    resp = post_align(src, delayed(src, period // 2))
    assert resp.status_code == 200
    body = resp.json()
    assert body["offset_samples"] == -(period // 2)
    assert body["correlation"] == pytest.approx(1.0)


def test_uncorrelated_tracks_return_422():
    left = make_source(seed=7)
    right = make_source(seed=8)
    assert post_align(left, right).status_code == 422


def test_constant_interval_returns_422():
    src = make_source(seed=9)
    left = src.copy()
    left[INTERVAL[0] : INTERVAL[1]] = 1000  # zero variance on the left track
    assert post_align(left, src).status_code == 422


def test_silent_interval_returns_422():
    src = make_source(seed=10)
    quiet = src.copy()
    quiet[INTERVAL[0] : INTERVAL[1]] = 0
    assert post_align(quiet, quiet.copy()).status_code == 422


def test_mono_wav_returns_422():
    src = make_source(seed=11)
    assert post_align(src, mono=True).status_code == 422


@pytest.mark.parametrize(
    "interval",
    [
        [0, RATE // 5 - 1],        # 1 frame shorter than 0.2 s
        [0, 2 * RATE + 1],         # 1 frame longer than 2 s
        [-1, 9600],                # negative start
        [N - 9600, N + 1],         # end out of bounds
        [INTERVAL[0], INTERVAL[0]],  # empty
        [INTERVAL[1], INTERVAL[0]],  # reversed
        [N, N],                    # empty at the tail
    ],
)
def test_interval_geometry_violations_return_422(interval):
    src = make_source(seed=12)
    assert post_align(src, delayed(src, 5), interval=interval).status_code == 422


@pytest.mark.parametrize("length", [RATE // 5, 2 * RATE])  # 0.2 s and 2 s
def test_interval_length_bounds_accepted(length):
    src = make_source(seed=13)
    resp = post_align(src, delayed(src, 11), interval=[4800, 4800 + length])
    assert resp.status_code == 200
    assert resp.json()["offset_samples"] == 11


def test_invalid_base64_returns_422():
    resp = client.post(
        "/v1/align-stereo",
        json={
            "wav_base64": "!!!not-base64!!!",
            "calibration_interval": INTERVAL,
        },
    )
    assert resp.status_code == 422


def test_garbage_wav_returns_422():
    resp = client.post(
        "/v1/align-stereo",
        json={
            "wav_base64": base64.b64encode(b"not a wav").decode(),
            "calibration_interval": INTERVAL,
        },
    )
    assert resp.status_code == 422


def test_non_compliant_wav_returns_422():
    src = make_source(seed=14)
    assert post_align(src, src.copy(), rate=44100).status_code == 422


@pytest.mark.parametrize("field", ["wav_base64", "calibration_interval"])
def test_missing_field_returns_422(field):
    src = make_source(seed=15)
    payload = {
        "wav_base64": stereo_wav_b64(src, src.copy()),
        "calibration_interval": INTERVAL,
    }
    del payload[field]
    assert client.post("/v1/align-stereo", json=payload).status_code == 422
