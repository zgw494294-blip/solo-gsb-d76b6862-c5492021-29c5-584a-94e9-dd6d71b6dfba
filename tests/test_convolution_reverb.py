import base64
import io
import json
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.signal import fftconvolve

from app.main import app

client = TestClient(app)

RATE = 48000
N = 2 * RATE  # 2 s source
M = RATE  # 1 s impulse response


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


def impulse_ir(amplitude=16384, delay=0, m=M):
    ir = np.zeros(m, dtype="<i2")
    ir[delay] = amplitude
    return ir


def decay_ir(m=M, amp=0.6):
    rng = np.random.default_rng(0)
    t = np.arange(m) / RATE
    return np.round(amp * np.exp(-t / 0.15) * rng.standard_normal(m) * 32768).astype(
        "<i2"
    )


def post_reverb(pcm, ir, wet=1.0, channels=1, ir_channels=1, rate=RATE,
                ir_rate=RATE):
    payload = {
        "wav_base64": wav_b64(pcm, channels, rate),
        "ir_wav_base64": wav_b64(ir, ir_channels, ir_rate),
        "wet": wet,
    }
    return client.post("/v1/convolution-reverb", json=payload)


def rms(x):
    return np.sqrt(np.mean(x.astype(np.float64) ** 2))


def test_wet_zero_returns_zero_padded_dry():
    pcm = sine(440, 0.2)
    resp = post_reverb(pcm, impulse_ir(), wet=0.0)
    assert resp.status_code == 200
    data = resp.json()
    assert data["output_frames"] == N + M - 1
    assert data["post_gain_db"] == 0.0
    params, out = decode_wav(data["wav_base64"])
    assert params == (1, 2, RATE)
    assert np.array_equal(out[:N], pcm)  # dry part bit-exact
    assert np.all(out[N:] == 0)  # tail is zero padding


def test_impulse_ir_wet_one_delays_and_scales():
    pcm = sine(440, 0.2)
    ir = impulse_ir(amplitude=16384, delay=100)  # 0.5 at frame 100
    resp = post_reverb(pcm, ir, wet=1.0)
    assert resp.status_code == 200
    data = resp.json()
    assert data["post_gain_db"] == 0.0
    _, out = decode_wav(data["wav_base64"])
    expected = np.zeros(N + M - 1, dtype=np.int64)
    expected[100 : 100 + N] = np.round(0.5 * pcm.astype(np.float64))
    assert np.abs(out.astype(np.int64) - expected).max() <= 1


def test_output_frames_and_reverb_tail():
    pcm = sine(440, 0.3)
    ir = decay_ir(m=RATE + RATE // 2)  # 1.5 s
    resp = post_reverb(pcm, ir, wet=0.5)
    assert resp.status_code == 200
    data = resp.json()
    assert data["output_frames"] == N + (RATE + RATE // 2) - 1
    _, out = decode_wav(data["wav_base64"])
    assert out.size == data["output_frames"]
    assert np.any(out[N:])  # the tail extends past the dry signal


def test_mono_ir_applies_to_each_stereo_channel():
    left = sine(440, 0.1)
    right = sine(550, 0.08)
    pcm = np.stack([left, right], axis=1)
    resp = post_reverb(pcm, impulse_ir(amplitude=16384), wet=1.0, channels=2)
    assert resp.status_code == 200
    params, out = decode_wav(resp.json()["wav_base64"])
    assert params == (2, 2, RATE)
    out = out.reshape(-1, 2)
    assert np.abs(out[:N, 0].astype(np.int64) - np.round(0.5 * left)).max() <= 1
    assert np.abs(out[:N, 1].astype(np.int64) - np.round(0.5 * right)).max() <= 1


def test_stereo_ir_with_mono_source_duplicates_dry():
    pcm = sine(440, 0.2)
    ir = np.stack([impulse_ir(16384), impulse_ir(8192, delay=50)], axis=1)
    resp = post_reverb(pcm, ir, wet=0.0, ir_channels=2)
    assert resp.status_code == 200
    params, out = decode_wav(resp.json()["wav_base64"])
    assert params == (2, 2, RATE)  # stereo IR widens the mono source
    out = out.reshape(-1, 2)
    assert np.array_equal(out[:N, 0], pcm)
    assert np.array_equal(out[:N, 1], pcm)
    assert np.all(out[N:] == 0)


def test_stereo_ir_with_mono_source_convolves_each_ir_channel():
    pcm = sine(440, 0.2)
    ir = np.stack([impulse_ir(16384), impulse_ir(8192, delay=50)], axis=1)
    resp = post_reverb(pcm, ir, wet=1.0, ir_channels=2)
    assert resp.status_code == 200
    _, out = decode_wav(resp.json()["wav_base64"])
    out = out.reshape(-1, 2)
    expected_l = np.round(0.5 * pcm.astype(np.float64))
    expected_r = np.zeros(N + M - 1, dtype=np.int64)
    expected_r[50 : 50 + N] = np.round(0.25 * pcm.astype(np.float64))
    assert np.abs(out[:N, 0].astype(np.int64) - expected_l).max() <= 1
    assert np.abs(out[:, 1].astype(np.int64) - expected_r).max() <= 1


def test_stereo_ir_with_stereo_source_convolves_per_channel():
    left = sine(440, 0.1)
    right = sine(550, 0.1)
    pcm = np.stack([left, right], axis=1)
    ir = np.stack([impulse_ir(16384), impulse_ir(8192)], axis=1)
    resp = post_reverb(pcm, ir, wet=1.0, channels=2, ir_channels=2)
    assert resp.status_code == 200
    _, out = decode_wav(resp.json()["wav_base64"])
    out = out.reshape(-1, 2)
    # Left channel only sees the left IR (0.5), right only the right (0.25).
    assert np.abs(out[:N, 0].astype(np.int64) - np.round(0.5 * left)).max() <= 1
    assert np.abs(out[:N, 1].astype(np.int64) - np.round(0.25 * right)).max() <= 1


def test_wet_mix_ratio():
    pcm = sine(440, 0.2)
    resp = post_reverb(pcm, impulse_ir(amplitude=16384), wet=0.5)
    assert resp.status_code == 200
    _, out = decode_wav(resp.json()["wav_base64"])
    # (1 - 0.5) * dry + 0.5 * (0.5 * dry) = 0.75 * dry
    expected = np.round(0.75 * pcm.astype(np.float64))
    assert np.abs(out[:N].astype(np.int64) - expected).max() <= 1


def test_overflow_applies_shared_non_positive_gain_without_clipping():
    # Constant DC convolved with a constant IR sums far past full scale.
    left = np.full(N, 29491, dtype="<i2")   # ~0.9
    right = np.full(N, 14746, dtype="<i2")  # ~0.45
    pcm = np.stack([left, right], axis=1)
    ir = np.full(M, 16384, dtype="<i2")     # 0.5 for 1 s
    resp = post_reverb(pcm, ir, wet=1.0, channels=2)
    assert resp.status_code == 200
    data = resp.json()
    assert data["post_gain_db"] < 0.0
    params, out = decode_wav(data["wav_base64"])
    assert params == (2, 2, RATE)
    out = out.reshape(-1, 2).astype(np.int64)
    assert np.max(np.abs(out)) == 32767  # pulled back to full scale, not clipped
    # One shared gain: the 2:1 channel ratio is preserved in the steady part.
    steady = out[RATE // 2 : N]
    assert np.abs(steady[:, 0] - 2 * steady[:, 1]).max() <= 2
    # The delivered PCM matches the mix scaled by the reported gain.
    mix_l = fftconvolve(left.astype(np.float64), ir.astype(np.float64), "full")
    mix_r = fftconvolve(right.astype(np.float64), ir.astype(np.float64), "full")
    mix = np.stack([mix_l, mix_r], axis=1) / 32768.0 / 32768.0
    gain = 10.0 ** (data["post_gain_db"] / 20.0)
    expected = np.round(mix * gain * 32768)
    assert np.abs(out - expected.astype(np.int64)).max() <= 3


def test_no_attenuation_when_mix_stays_in_range():
    pcm = sine(440, 0.2)
    resp = post_reverb(pcm, decay_ir(amp=0.02), wet=0.7)
    assert resp.status_code == 200
    assert resp.json()["post_gain_db"] == 0.0


def test_silent_source_passes_through():
    pcm = np.zeros(N, dtype="<i2")
    resp = post_reverb(pcm, decay_ir(), wet=0.8)
    assert resp.status_code == 200
    data = resp.json()
    assert data["post_gain_db"] == 0.0
    _, out = decode_wav(data["wav_base64"])
    assert np.all(out == 0)


def test_ir_duration_bounds():
    pcm = sine(440, 0.1)
    assert post_reverb(pcm, impulse_ir(m=RATE)).status_code == 200        # 1.0 s
    assert post_reverb(pcm, impulse_ir(m=2 * RATE)).status_code == 200  # 2.0 s
    assert post_reverb(pcm, impulse_ir(m=RATE - 1)).status_code == 422
    assert post_reverb(pcm, impulse_ir(m=2 * RATE + 1)).status_code == 422


def test_all_silent_ir_returns_422():
    pcm = sine(440, 0.1)
    assert post_reverb(pcm, np.zeros(M, dtype="<i2")).status_code == 422


def test_single_nonzero_ir_sample_accepted():
    pcm = sine(440, 0.1)
    ir = np.zeros(M, dtype="<i2")
    ir[M - 1] = 1
    assert post_reverb(pcm, ir).status_code == 200


def test_source_duration_bounds_enforced():
    ir = impulse_ir()
    assert post_reverb(sine(440, 0.1, n=RATE // 2), ir).status_code == 422
    assert post_reverb(sine(440, 0.1, n=11 * RATE), ir).status_code == 422


def test_wrong_sample_rates_return_422():
    pcm = sine(440, 0.1)
    ir = impulse_ir()
    assert post_reverb(pcm, ir, rate=44100).status_code == 422
    assert post_reverb(pcm, ir, ir_rate=44100).status_code == 422


def test_invalid_base64_returns_422():
    payload = {
        "wav_base64": "!!!not-base64!!!",
        "ir_wav_base64": wav_b64(impulse_ir()),
        "wet": 0.5,
    }
    assert client.post("/v1/convolution-reverb", json=payload).status_code == 422
    payload = {
        "wav_base64": wav_b64(sine(440, 0.1)),
        "ir_wav_base64": "!!!not-base64!!!",
        "wet": 0.5,
    }
    assert client.post("/v1/convolution-reverb", json=payload).status_code == 422


def test_garbage_wav_returns_422():
    payload = {
        "wav_base64": base64.b64encode(b"not a wav").decode(),
        "ir_wav_base64": wav_b64(impulse_ir()),
        "wet": 0.5,
    }
    assert client.post("/v1/convolution-reverb", json=payload).status_code == 422
    payload = {
        "wav_base64": wav_b64(sine(440, 0.1)),
        "ir_wav_base64": base64.b64encode(b"not a wav").decode(),
        "wet": 0.5,
    }
    assert client.post("/v1/convolution-reverb", json=payload).status_code == 422


@pytest.mark.parametrize("wet", [-0.1, 1.1])
def test_wet_out_of_range_returns_422(wet):
    assert post_reverb(sine(440, 0.1), impulse_ir(), wet=wet).status_code == 422


@pytest.mark.parametrize("wet", [0.0, 1.0])
def test_wet_bounds_accepted(wet):
    assert post_reverb(sine(440, 0.1), impulse_ir(), wet=wet).status_code == 200


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_wet_returns_422(bad):
    payload = {
        "wav_base64": wav_b64(sine(440, 0.1)),
        "ir_wav_base64": wav_b64(impulse_ir()),
        "wet": bad,
    }
    resp = client.post(
        "/v1/convolution-reverb",
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


@pytest.mark.parametrize("field", ["wav_base64", "ir_wav_base64", "wet"])
def test_missing_field_returns_422(field):
    payload = {
        "wav_base64": wav_b64(sine(440, 0.1)),
        "ir_wav_base64": wav_b64(impulse_ir()),
        "wet": 0.5,
    }
    del payload[field]
    assert client.post("/v1/convolution-reverb", json=payload).status_code == 422
