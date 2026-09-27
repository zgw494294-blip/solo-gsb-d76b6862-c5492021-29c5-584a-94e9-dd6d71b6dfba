import base64
import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

RATE = 48000
N = 2 * RATE  # 2 s of audio
REF_END = RATE // 2  # noise-only reference: first 0.5 s


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


def make_noisy_sine(n=N, freq=440.0, amp=0.4, noise_amp=0.05, seed=1):
    """Sine plus stationary noise; the first REF_END samples are noise only."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / RATE
    noise = noise_amp * rng.standard_normal(n)
    clean = amp * np.sin(2 * np.pi * freq * t)
    sig = clean + noise
    sig[:REF_END] = noise[:REF_END]
    pcm = np.round(sig * 32768).astype("<i2")
    return pcm, np.round(clean * 32768).astype("<i2")


def post_denoise(pcm, interval=None, strength=1.0, channels=1, rate=RATE):
    if interval is None:
        interval = [0, REF_END]
    return client.post(
        "/v1/denoise",
        json={
            "wav_base64": wav_b64(pcm, channels, rate),
            "noise_interval": interval,
            "strength": strength,
        },
    )


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def test_strength_zero_returns_pcm_unchanged():
    pcm, _ = make_noisy_sine()
    resp = post_denoise(pcm, strength=0.0)
    assert resp.status_code == 200
    params, out = decode_wav(resp.json()["wav_base64"])
    assert params == (1, 2, RATE)
    assert out.size == N
    assert np.array_equal(out, pcm)


def test_strength_zero_returns_pcm_unchanged_stereo():
    left, _ = make_noisy_sine(seed=3)
    right, _ = make_noisy_sine(freq=550.0, seed=4)
    pcm = np.stack([left, right], axis=1)
    resp = post_denoise(pcm, strength=0.0, channels=2)
    assert resp.status_code == 200
    params, out = decode_wav(resp.json()["wav_base64"])
    assert params == (2, 2, RATE)
    assert np.array_equal(out.reshape(-1, 2), pcm)


def test_denoise_suppresses_noise_only_region():
    pcm, _ = make_noisy_sine()
    resp = post_denoise(pcm, strength=1.0)
    assert resp.status_code == 200
    _, out = decode_wav(resp.json()["wav_base64"])
    assert out.size == N
    # The noise-only reference region is pushed far down.
    assert rms(out[:REF_END]) < 0.75 * rms(pcm[:REF_END])


def test_denoise_improves_snr_in_signal_region():
    pcm, clean = make_noisy_sine()
    resp = post_denoise(pcm, strength=1.0)
    assert resp.status_code == 200
    _, out = decode_wav(resp.json()["wav_base64"])
    # Skip the STFT transition around the reference boundary.
    region = slice(REF_END + 2048, N)
    err_in = rms(pcm[region] - clean[region])
    err_out = rms(out[region] - clean[region])
    assert err_out < 0.75 * err_in
    # The desired signal itself survives: output stays close to the sine.
    assert rms(out[region] - clean[region]) < 0.3 * rms(clean[region])


def test_stronger_strength_suppresses_more_noise():
    pcm, _ = make_noisy_sine()
    _, out_half = decode_wav(
        post_denoise(pcm, strength=0.5).json()["wav_base64"]
    )
    _, out_full = decode_wav(
        post_denoise(pcm, strength=1.0).json()["wav_base64"]
    )
    assert rms(out_full[:REF_END]) < rms(out_half[:REF_END])


def test_stereo_channels_denoised_independently():
    left, _ = make_noisy_sine(seed=5)
    right, _ = make_noisy_sine(freq=660.0, seed=6)
    pcm = np.stack([left, right], axis=1)
    resp = post_denoise(pcm, strength=1.0, channels=2)
    assert resp.status_code == 200
    params, out = decode_wav(resp.json()["wav_base64"])
    assert params == (2, 2, RATE)
    out = out.reshape(-1, 2)
    assert out.shape == (N, 2)
    for ch in range(2):
        assert rms(out[:REF_END, ch]) < 0.75 * rms(pcm[:REF_END, ch])


def test_reference_at_file_tail_accepted():
    pcm, _ = make_noisy_sine()
    # Move the noise-only reference to the last 0.5 s.
    pcm[REF_END:] = pcm[: N - REF_END]
    resp = post_denoise(pcm, interval=[N - REF_END, N], strength=1.0)
    assert resp.status_code == 200
    _, out = decode_wav(resp.json()["wav_base64"])
    assert rms(out[N - REF_END :]) < 0.75 * rms(pcm[N - REF_END :])


@pytest.mark.parametrize("length", [RATE // 5, 2 * RATE])  # 0.2 s and 2 s
def test_reference_length_bounds_accepted(length):
    rng = np.random.default_rng(0)
    pcm = np.round(0.05 * rng.standard_normal(N) * 32768).astype("<i2")
    resp = post_denoise(pcm, interval=[0, length], strength=0.5)
    assert resp.status_code == 200


@pytest.mark.parametrize(
    "interval",
    [
        [0, RATE // 5 - 1],        # 1 frame shorter than 0.2 s
        [0, 2 * RATE + 1],         # 1 frame longer than 2 s
        [-1, REF_END],             # negative start
        [N - REF_END, N + 1],      # end out of bounds
        [REF_END, REF_END],        # empty
        [REF_END + 100, REF_END],  # reversed
        [N, N],                    # empty at the tail
    ],
)
def test_interval_geometry_violations_return_422(interval):
    pcm, _ = make_noisy_sine()
    assert post_denoise(pcm, interval=interval).status_code == 422


def test_all_zero_reference_returns_422():
    pcm, _ = make_noisy_sine()
    pcm[:REF_END] = 0
    assert post_denoise(pcm).status_code == 422


def test_reference_zero_on_one_stereo_channel_returns_422():
    left, _ = make_noisy_sine(seed=7)
    right, _ = make_noisy_sine(freq=660.0, seed=8)
    pcm = np.stack([left, right], axis=1)
    pcm[:REF_END, 1] = 0  # right channel silent inside the reference
    assert post_denoise(pcm, channels=2).status_code == 422


@pytest.mark.parametrize("strength", [-0.1, 1.1])
def test_strength_out_of_range_returns_422(strength):
    pcm, _ = make_noisy_sine()
    assert post_denoise(pcm, strength=strength).status_code == 422


def test_invalid_base64_returns_422():
    resp = client.post(
        "/v1/denoise",
        json={
            "wav_base64": "!!!not-base64!!!",
            "noise_interval": [0, REF_END],
            "strength": 0.5,
        },
    )
    assert resp.status_code == 422


def test_garbage_wav_returns_422():
    resp = client.post(
        "/v1/denoise",
        json={
            "wav_base64": base64.b64encode(b"not a wav").decode(),
            "noise_interval": [0, REF_END],
            "strength": 0.5,
        },
    )
    assert resp.status_code == 422


def test_non_compliant_wav_returns_422():
    pcm, _ = make_noisy_sine()
    assert post_denoise(pcm, rate=44100).status_code == 422


@pytest.mark.parametrize(
    "field", ["wav_base64", "noise_interval", "strength"]
)
def test_missing_field_returns_422(field):
    pcm, _ = make_noisy_sine()
    payload = {
        "wav_base64": wav_b64(pcm),
        "noise_interval": [0, REF_END],
        "strength": 0.5,
    }
    del payload[field]
    assert client.post("/v1/denoise", json=payload).status_code == 422
