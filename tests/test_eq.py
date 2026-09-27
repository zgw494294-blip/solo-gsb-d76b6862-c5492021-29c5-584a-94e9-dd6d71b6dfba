import base64
import io
import json
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

RATE = 48000
N = 2 * RATE  # 2 s of audio

BAND = {"frequency_hz": 1000.0, "q": 1.0, "gain_db": 6.0}


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


def sine(freq, amp, n=N):
    t = np.arange(n) / RATE
    return np.round(amp * np.sin(2 * np.pi * freq * t) * 32768).astype("<i2")


def post_eq(pcm, channels=1, rate=RATE, bands=None, ceiling_dbtp=-1.0):
    payload = {
        "wav_base64": wav_b64(pcm, channels, rate),
        "bands": [BAND] if bands is None else bands,
        "ceiling_dbtp": ceiling_dbtp,
    }
    return client.post("/v1/eq", json=payload)


def post_eq_raw(pcm, bands=None, ceiling_dbtp=-1.0):
    # Raw body so non-finite floats (NaN/Infinity, allowed by Python's json
    # encoder and decoder) actually reach the server.
    payload = {
        "wav_base64": wav_b64(pcm),
        "bands": [BAND] if bands is None else bands,
        "ceiling_dbtp": ceiling_dbtp,
    }
    return client.post(
        "/v1/eq",
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )


def rms(x):
    return np.sqrt(np.mean(x.astype(np.float64) ** 2))


def test_zero_gain_bands_leave_pcm_unchanged():
    pcm = sine(440, 0.1)
    bands = [
        {"frequency_hz": 100.0, "q": 0.5, "gain_db": 0.0},
        {"frequency_hz": 1000.0, "q": 1.0, "gain_db": 0.0},
        {"frequency_hz": 10000.0, "q": 5.0, "gain_db": 0.0},
    ]
    resp = post_eq(pcm, bands=bands)
    assert resp.status_code == 200
    data = resp.json()
    assert data["post_gain_db"] == 0.0
    params, out = decode_wav(data["wav_base64"])
    assert params == (1, 2, RATE)
    assert np.array_equal(out, pcm)


def test_boost_gain_matches_rbj_at_centre():
    pcm = sine(1000, 0.05)
    resp = post_eq(pcm, bands=[{"frequency_hz": 1000.0, "q": 1.0, "gain_db": 6.0}])
    assert resp.status_code == 200
    data = resp.json()
    assert data["post_gain_db"] == 0.0
    _, out = decode_wav(data["wav_base64"])
    # Steady-state gain at the band centre is exactly 10^(gain_db/20).
    ratio = rms(out[RATE:]) / rms(pcm[RATE:])
    assert ratio == pytest.approx(10.0 ** (6.0 / 20.0), rel=0.01)


def test_cut_gain_matches_rbj_at_centre():
    pcm = sine(1000, 0.05)
    resp = post_eq(pcm, bands=[{"frequency_hz": 1000.0, "q": 2.0, "gain_db": -6.0}])
    assert resp.status_code == 200
    _, out = decode_wav(resp.json()["wav_base64"])
    ratio = rms(out[RATE:]) / rms(pcm[RATE:])
    assert ratio == pytest.approx(10.0 ** (-6.0 / 20.0), rel=0.01)


def test_off_band_signal_mostly_unaffected():
    pcm = sine(200, 0.05)
    resp = post_eq(pcm, bands=[{"frequency_hz": 5000.0, "q": 5.0, "gain_db": 12.0}])
    assert resp.status_code == 200
    _, out = decode_wav(resp.json()["wav_base64"])
    ratio = rms(out[RATE:]) / rms(pcm[RATE:])
    assert ratio == pytest.approx(1.0, rel=0.02)


def test_opposite_bands_cancel():
    pcm = sine(997, 0.1)
    bands = [
        {"frequency_hz": 1000.0, "q": 2.0, "gain_db": 6.0},
        {"frequency_hz": 1000.0, "q": 2.0, "gain_db": -6.0},
    ]
    resp = post_eq(pcm, bands=bands)
    assert resp.status_code == 200
    _, out = decode_wav(resp.json()["wav_base64"])
    assert np.max(np.abs(out.astype(np.int64) - pcm.astype(np.int64))) <= 1


def test_ceiling_binds_after_boost():
    # 0.4 amp sine boosted +12 dB lands near +4 dBTP: the ceiling must
    # pull it back down with a single non-positive gain, without clipping.
    pcm = sine(1000, 0.4)
    resp = post_eq(
        pcm,
        bands=[{"frequency_hz": 1000.0, "q": 1.0, "gain_db": 12.0}],
        ceiling_dbtp=-1.0,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["post_gain_db"] < 0.0
    assert data["output_true_peak_dbtp"] <= -1.0 + 1e-3
    assert data["output_true_peak_dbtp"] > -1.5  # not backed off endlessly
    _, out = decode_wav(data["wav_base64"])
    assert np.max(np.abs(out.astype(np.int64))) <= 32767


def test_post_gain_never_positive():
    # Filtered peak far below the ceiling: no make-up gain is applied.
    pcm = sine(1000, 0.01)
    resp = post_eq(
        pcm,
        bands=[{"frequency_hz": 1000.0, "q": 1.0, "gain_db": -12.0}],
        ceiling_dbtp=-0.5,
    )
    assert resp.status_code == 200
    assert resp.json()["post_gain_db"] == 0.0


def test_stereo_channels_filtered_independently():
    left = sine(440, 0.05)
    right = sine(8000, 0.05)
    pcm = np.stack([left, right], axis=1)
    resp = post_eq(
        pcm,
        channels=2,
        bands=[{"frequency_hz": 440.0, "q": 4.0, "gain_db": 12.0}],
    )
    assert resp.status_code == 200
    params, out = decode_wav(resp.json()["wav_base64"])
    assert params == (2, 2, RATE)
    out = out.reshape(-1, 2)
    left_ratio = rms(out[RATE:, 0]) / rms(left[RATE:])
    right_ratio = rms(out[RATE:, 1]) / rms(right[RATE:])
    assert left_ratio == pytest.approx(10.0 ** (12.0 / 20.0), rel=0.02)
    assert right_ratio == pytest.approx(1.0, rel=0.02)


def test_output_format_and_length_preserved():
    pcm = np.stack([sine(440, 0.05), sine(550, 0.04)], axis=1)
    resp = post_eq(pcm, channels=2)
    assert resp.status_code == 200
    params, out = decode_wav(resp.json()["wav_base64"])
    assert params == (2, 2, RATE)
    assert out.size == 2 * N  # interleaved stereo, same frame count


def test_response_peak_fields():
    pcm = sine(997, 0.1)
    resp = post_eq(pcm)
    assert resp.status_code == 200
    data = resp.json()
    assert data["input_true_peak_dbtp"] == pytest.approx(
        20.0 * np.log10(0.1), abs=0.01
    )
    # +6 dB at 1 kHz barely touches a 997 Hz sine: output peak rises ~6 dB.
    assert data["output_true_peak_dbtp"] == pytest.approx(
        data["input_true_peak_dbtp"] + 6.0, abs=0.2
    )


def test_silence_passes_through_with_null_peaks():
    pcm = np.zeros(RATE, dtype="<i2")
    resp = post_eq(pcm)
    assert resp.status_code == 200
    data = resp.json()
    assert data["input_true_peak_dbtp"] is None
    assert data["output_true_peak_dbtp"] is None
    assert data["post_gain_db"] == 0.0
    _, out = decode_wav(data["wav_base64"])
    assert np.array_equal(out, pcm)


def test_invalid_base64_returns_422():
    payload = {"wav_base64": "!!!not-base64!!!", "bands": [BAND], "ceiling_dbtp": -1.0}
    assert client.post("/v1/eq", json=payload).status_code == 422


def test_garbage_wav_returns_422():
    payload = {
        "wav_base64": base64.b64encode(b"not a wav").decode(),
        "bands": [BAND],
        "ceiling_dbtp": -1.0,
    }
    assert client.post("/v1/eq", json=payload).status_code == 422


def test_non_compliant_wav_returns_422():
    pcm = sine(440, 0.1)
    assert post_eq(pcm, rate=44100).status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [
        ("frequency_hz", 39.9),
        ("frequency_hz", 16000.1),
        ("q", 0.29),
        ("q", 10.1),
        ("gain_db", -12.1),
        ("gain_db", 12.1),
    ],
)
def test_band_param_out_of_range_returns_422(field, value):
    pcm = sine(440, 0.1)
    band = {**BAND, field: value}
    assert post_eq(pcm, bands=[band]).status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [
        ("frequency_hz", 40.0),
        ("frequency_hz", 16000.0),
        ("q", 0.3),
        ("q", 10.0),
        ("gain_db", -12.0),
        ("gain_db", 12.0),
    ],
)
def test_band_param_bounds_accepted(field, value):
    pcm = sine(440, 0.01)
    band = {**BAND, field: value}
    assert post_eq(pcm, bands=[band]).status_code == 200


@pytest.mark.parametrize("ceiling", [-6.1, -0.4])
def test_ceiling_out_of_range_returns_422(ceiling):
    pcm = sine(440, 0.1)
    assert post_eq(pcm, ceiling_dbtp=ceiling).status_code == 422


@pytest.mark.parametrize("ceiling", [-6.0, -0.5])
def test_ceiling_bounds_accepted(ceiling):
    pcm = sine(440, 0.01)
    assert post_eq(pcm, ceiling_dbtp=ceiling).status_code == 200


def test_band_count_limits():
    pcm = sine(440, 0.01)
    assert post_eq(pcm, bands=[]).status_code == 422
    assert post_eq(pcm, bands=[BAND] * 6).status_code == 200
    assert post_eq(pcm, bands=[BAND] * 7).status_code == 422


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["frequency_hz", "q", "gain_db"])
def test_non_finite_band_param_returns_422(field, bad):
    pcm = sine(440, 0.1)
    band = {**BAND, field: bad}
    assert post_eq_raw(pcm, bands=[band]).status_code == 422


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_ceiling_returns_422(bad):
    pcm = sine(440, 0.1)
    assert post_eq_raw(pcm, ceiling_dbtp=bad).status_code == 422


@pytest.mark.parametrize("field", ["wav_base64", "bands", "ceiling_dbtp"])
def test_missing_field_returns_422(field):
    pcm = sine(440, 0.1)
    payload = {"wav_base64": wav_b64(pcm), "bands": [BAND], "ceiling_dbtp": -1.0}
    del payload[field]
    assert client.post("/v1/eq", json=payload).status_code == 422


@pytest.mark.parametrize("field", ["frequency_hz", "q", "gain_db"])
def test_band_missing_field_returns_422(field):
    pcm = sine(440, 0.1)
    band = {k: v for k, v in BAND.items() if k != field}
    assert post_eq(pcm, bands=[band]).status_code == 422
