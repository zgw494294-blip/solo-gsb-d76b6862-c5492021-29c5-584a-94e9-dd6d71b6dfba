import base64
import io
import math
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

RATE = 48000
N = 2 * RATE  # 2 s of audio

DEFAULTS = {
    "threshold_dbfs": -20.0,
    "ratio": 4.0,
    "attack_ms": 1.0,
    "release_ms": 10.0,
    "knee_db": 0.0,
}


def wav_b64(pcm: np.ndarray, channels: int = 1, rate: int = RATE) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
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


def post_compress(pcm, channels=1, rate=RATE, **overrides):
    payload = {"wav_base64": wav_b64(pcm, channels, rate)}
    payload.update({**DEFAULTS, **overrides})
    return client.post("/v1/compress", json=payload)


def static_gain_db(level_db, threshold=-20.0, ratio=4.0):
    """Hard-knee target curve gain for a settled envelope level."""
    if level_db >= threshold:
        return threshold + (level_db - threshold) / ratio - level_db
    return 0.0


def test_ratio_one_returns_pcm_unchanged():
    t = np.arange(N) / RATE
    pcm = np.round(0.4 * np.sin(2 * np.pi * 440 * t) * 32768).astype("<i2")
    resp = post_compress(pcm, ratio=1.0)
    assert resp.status_code == 200
    params, out = decode_wav(resp.json()["wav_base64"])
    assert params == (1, 2, RATE)
    assert np.array_equal(out, pcm)


def test_ratio_one_returns_pcm_unchanged_stereo():
    t = np.arange(N) / RATE
    left = np.round(0.5 * np.sin(2 * np.pi * 440 * t) * 32768).astype("<i2")
    right = np.round(0.3 * np.sin(2 * np.pi * 660 * t) * 32768).astype("<i2")
    pcm = np.stack([left, right], axis=1)
    resp = post_compress(pcm, channels=2, ratio=1.0, knee_db=12.0)
    assert resp.status_code == 200
    params, out = decode_wav(resp.json()["wav_base64"])
    assert params == (2, 2, RATE)
    assert np.array_equal(out.reshape(-1, 2), pcm)


def test_constant_signal_matches_static_curve():
    pcm = np.full(N, 16384, dtype="<i2")  # 0.5 full scale => -6.02 dBFS
    resp = post_compress(pcm, threshold_dbfs=-20.0, ratio=4.0, knee_db=0.0)
    assert resp.status_code == 200
    params, out = decode_wav(resp.json()["wav_base64"])
    assert params == (1, 2, RATE)
    assert out.size == N
    level = 20.0 * math.log10(0.5)
    gain_db = static_gain_db(level)
    expected = round(0.5 * 10.0 ** (gain_db / 20.0) * 32768)
    tail = out[RATE:].astype(np.int64)  # envelope settled in the second half
    assert np.all(np.abs(tail - expected) <= 2)


@pytest.mark.parametrize("knee_db", [0.0, 12.0])
def test_below_threshold_signal_unchanged(knee_db):
    pcm = np.full(N, 328, dtype="<i2")  # ~ -40 dBFS, below threshold and knee
    resp = post_compress(pcm, threshold_dbfs=-20.0, ratio=4.0, knee_db=knee_db)
    assert resp.status_code == 200
    _, out = decode_wav(resp.json()["wav_base64"])
    assert np.array_equal(out, pcm)


def test_stereo_channels_share_one_gain():
    # Left at -6 dBFS (over threshold), right at -26 dBFS (below threshold):
    # the linked envelope still attenuates both channels by the same gain.
    left = np.full(N, 16384, dtype="<i2")
    right = np.full(N, 1638, dtype="<i2")
    pcm = np.stack([left, right], axis=1)
    resp = post_compress(pcm, channels=2, threshold_dbfs=-20.0, ratio=4.0)
    assert resp.status_code == 200
    params, out = decode_wav(resp.json()["wav_base64"])
    assert params == (2, 2, RATE)
    tail = out.reshape(-1, 2)[RATE:].astype(np.int64)
    gain_db = static_gain_db(20.0 * math.log10(0.5))
    gain = 10.0 ** (gain_db / 20.0)
    assert np.all(np.abs(tail[:, 0] - round(16384 * gain)) <= 2)
    assert np.all(np.abs(tail[:, 1] - round(1638 * gain)) <= 2)


def test_soft_knee_partial_attenuation():
    pcm = np.full(N, 3277, dtype="<i2")  # ~ -20 dBFS: centre of a 12 dB knee
    level = 20.0 * math.log10(3277 / 32768.0)

    hard = post_compress(pcm, threshold_dbfs=-20.0, ratio=4.0, knee_db=0.0)
    assert hard.status_code == 200
    _, hard_out = decode_wav(hard.json()["wav_base64"])
    # At the threshold the hard knee applies (almost) no gain.
    assert np.array_equal(hard_out, pcm)

    soft = post_compress(pcm, threshold_dbfs=-20.0, ratio=4.0, knee_db=12.0)
    assert soft.status_code == 200
    _, soft_out = decode_wav(soft.json()["wav_base64"])
    x = level - (-20.0) + 6.0
    gain_db = (1.0 / 4.0 - 1.0) * x * x / (2.0 * 12.0)
    expected = round(3277 * 10.0 ** (gain_db / 20.0))
    tail = soft_out[RATE:].astype(np.int64)
    assert expected < 3277  # some attenuation, unlike the hard knee
    assert np.all(np.abs(tail - expected) <= 2)


def test_attack_delays_compression():
    pcm = np.concatenate(
        [np.zeros(RATE, dtype="<i2"), np.full(RATE, 16384, dtype="<i2")]
    )
    resp = post_compress(pcm, attack_ms=50.0, release_ms=10.0)
    assert resp.status_code == 200
    _, out = decode_wav(resp.json()["wav_base64"])
    onset = RATE
    # The envelope starts at 0, so the onset passes unattenuated at first.
    assert out[onset] == 16384
    assert np.max(np.abs(out[onset : onset + 480].astype(np.int64))) >= 16000
    # Once settled (well within the second half), the static curve applies.
    gain_db = static_gain_db(20.0 * math.log10(0.5))
    expected = round(0.5 * 10.0 ** (gain_db / 20.0) * 32768)
    tail = out[onset + RATE // 2 :].astype(np.int64)
    assert np.all(np.abs(tail - expected) <= 2)


def test_release_recovers_after_loud_section():
    pcm = np.concatenate(
        [np.full(RATE, 16384, dtype="<i2"), np.full(RATE, 328, dtype="<i2")]
    )
    resp = post_compress(pcm, attack_ms=1.0, release_ms=200.0)
    assert resp.status_code == 200
    _, out = decode_wav(resp.json()["wav_base64"])
    quiet = out[RATE:]
    # Just after the transition the envelope is still high: the quiet
    # section is compressed down (328 -> ~100) before the gain recovers.
    assert np.max(np.abs(quiet[:480].astype(np.int64))) < 150
    # After the release time the quiet signal passes untouched.
    assert np.all(quiet[-RATE // 2 :] == 328)


def test_compressor_never_boosts():
    t = np.arange(N) / RATE
    pcm = np.round(0.9 * np.sin(2 * np.pi * 440 * t) * 32768).astype("<i2")
    resp = post_compress(pcm, threshold_dbfs=-30.0, ratio=10.0, knee_db=12.0)
    assert resp.status_code == 200
    _, out = decode_wav(resp.json()["wav_base64"])
    assert np.max(np.abs(out.astype(np.int64))) <= np.max(
        np.abs(pcm.astype(np.int64))
    )
    assert np.sqrt(np.mean(out.astype(np.float64) ** 2)) < 0.8 * np.sqrt(
        np.mean(pcm.astype(np.float64) ** 2)
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("threshold_dbfs", -40.1),
        ("threshold_dbfs", -5.9),
        ("ratio", 0.9),
        ("ratio", 20.1),
        ("attack_ms", 0.9),
        ("attack_ms", 50.1),
        ("release_ms", 9.9),
        ("release_ms", 500.1),
        ("knee_db", -0.1),
        ("knee_db", 12.1),
    ],
)
def test_param_out_of_range_returns_422(field, value):
    pcm = np.full(RATE, 1000, dtype="<i2")
    assert post_compress(pcm, **{field: value}).status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [
        ("threshold_dbfs", -40.0),
        ("threshold_dbfs", -6.0),
        ("ratio", 1.0),
        ("ratio", 20.0),
        ("attack_ms", 1.0),
        ("attack_ms", 50.0),
        ("release_ms", 10.0),
        ("release_ms", 500.0),
        ("knee_db", 0.0),
        ("knee_db", 12.0),
    ],
)
def test_param_bounds_accepted(field, value):
    pcm = np.full(RATE, 1000, dtype="<i2")
    assert post_compress(pcm, **{field: value}).status_code == 200


def test_invalid_base64_returns_422():
    payload = {**DEFAULTS, "wav_base64": "!!!not-base64!!!"}
    assert client.post("/v1/compress", json=payload).status_code == 422


def test_garbage_wav_returns_422():
    payload = {**DEFAULTS, "wav_base64": base64.b64encode(b"not a wav").decode()}
    assert client.post("/v1/compress", json=payload).status_code == 422


def test_non_compliant_wav_returns_422():
    pcm = np.full(N, 1000, dtype="<i2")
    assert post_compress(pcm, rate=44100).status_code == 422


@pytest.mark.parametrize("field", list(DEFAULTS) + ["wav_base64"])
def test_missing_field_returns_422(field):
    pcm = np.full(RATE, 1000, dtype="<i2")
    payload = {"wav_base64": wav_b64(pcm), **DEFAULTS}
    del payload[field]
    assert client.post("/v1/compress", json=payload).status_code == 422
