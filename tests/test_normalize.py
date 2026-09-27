import base64
import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

RATE = 48000


def make_wav_base64(
    freq=997.0, duration=2.0, amp=0.1, channels=1, rate=RATE, silence=False
) -> str:
    t = np.arange(int(rate * duration)) / rate
    sig = np.zeros_like(t) if silence else amp * np.sin(2 * np.pi * freq * t)
    if channels == 2:
        sig = np.stack([sig, sig], axis=1)
    pcm = np.clip(np.round(sig * 32768), -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def post(wav_b64, target_lufs=-23.0, ceiling_dbtp=-1.0):
    return client.post(
        "/v1/normalize",
        json={
            "wav_base64": wav_b64,
            "target_lufs": target_lufs,
            "ceiling_dbtp": ceiling_dbtp,
        },
    )


def decode_wav(b64):
    raw = base64.b64decode(b64)
    with wave.open(io.BytesIO(raw), "rb") as wf:
        params = (wf.getnchannels(), wf.getsampwidth(), wf.getframerate())
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    return params, pcm


def test_normalize_mono_reaches_target():
    resp = post(make_wav_base64(), target_lufs=-23.0, ceiling_dbtp=-1.0)
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_reached"] is True
    assert abs(data["output_loudness_lufs"] - (-23.0)) <= 0.1
    assert data["output_true_peak_dbtp"] <= -1.0
    # gain = target - input, and input peak stays well under the ceiling
    assert data["gain_db"] == pytest.approx(
        -23.0 - data["input_loudness_lufs"], abs=0.01
    )
    (ch, width, rate), pcm = decode_wav(data["wav_base64"])
    # same format back: mono, 16-bit, 48 kHz, same length
    assert (ch, width, rate) == (1, 2, RATE)
    assert len(pcm) == 2 * RATE


def test_normalize_stereo():
    resp = post(make_wav_base64(channels=2, amp=0.05), target_lufs=-18.0)
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_reached"] is True
    (ch, width, rate), pcm = decode_wav(data["wav_base64"])
    assert (ch, width, rate) == (2, 2, RATE)
    assert len(pcm) == 2 * 2 * RATE  # interleaved


def test_ceiling_limits_gain_and_target_missed():
    # High crest factor: quiet bed + loud 50 ms burst, so the true-peak
    # ceiling binds long before the loudness target is reached.
    t = np.arange(2 * RATE) / RATE
    sig = 0.05 * np.sin(2 * np.pi * 997 * t)
    burst = slice(RATE, RATE + RATE // 20)
    sig[burst] = 0.95 * np.sin(2 * np.pi * 3000 * t[burst])
    pcm = np.round(sig * 32768).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(pcm.tobytes())
    wav_b64 = base64.b64encode(buf.getvalue()).decode()

    resp = post(wav_b64, target_lufs=-10.0, ceiling_dbtp=-6.0)
    assert resp.status_code == 200
    data = resp.json()
    assert data["output_true_peak_dbtp"] <= -6.0
    # gain is the ceiling-limited value (minus any tiny backoff), not the
    # much larger gain the loudness target would need
    assert data["gain_db"] == pytest.approx(
        -6.0 - data["input_true_peak_dbtp"], abs=0.01
    )
    assert data["target_reached"] is False
    assert data["output_loudness_lufs"] < -10.1


def test_attenuation_when_input_too_loud():
    resp = post(make_wav_base64(amp=0.9), target_lufs=-30.0, ceiling_dbtp=-0.5)
    assert resp.status_code == 200
    data = resp.json()
    assert data["gain_db"] < 0
    assert data["target_reached"] is True


def test_silence_returns_422():
    resp = post(make_wav_base64(silence=True))
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rate": 44100},          # wrong sample rate
        {"duration": 0.5},        # too short
        {"duration": 10.5},       # too long
    ],
)
def test_non_compliant_wav_returns_422(kwargs):
    assert post(make_wav_base64(**kwargs)).status_code == 422


def test_8bit_wav_returns_422():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(RATE)
        wf.writeframes(bytes([128] * RATE * 2))
    assert post(base64.b64encode(buf.getvalue()).decode()).status_code == 422


def test_invalid_base64_returns_422():
    assert post("!!!not-base64!!!").status_code == 422


def test_garbage_wav_returns_422():
    assert post(base64.b64encode(b"not a wav file at all").decode()).status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {"target_lufs": -31.0, "ceiling_dbtp": -1.0},   # target too low
        {"target_lufs": -9.0, "ceiling_dbtp": -1.0},    # target too high
        {"target_lufs": -23.0, "ceiling_dbtp": -6.5},   # ceiling too low
        {"target_lufs": -23.0, "ceiling_dbtp": -0.4},   # ceiling too high
    ],
)
def test_out_of_range_params_return_422(payload):
    resp = client.post(
        "/v1/normalize", json={"wav_base64": make_wav_base64(), **payload}
    )
    assert resp.status_code == 422


def test_health():
    assert client.get("/health").json() == {"status": "ok"}
