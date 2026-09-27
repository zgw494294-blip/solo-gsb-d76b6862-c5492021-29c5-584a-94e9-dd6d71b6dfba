import base64
import io
import json
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.onsets import _is_local_peak

client = TestClient(app)

RATE = 48000
N = 2 * RATE  # 2 s of audio
STRONG = 26214  # ~0.8 full scale
WEAK = 6554  # ~0.2 full scale


def wav_b64(pcm: np.ndarray, channels: int = 1, rate: int = RATE) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.astype("<i2").tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def clicks(positions, n=N):
    """Silence with a single-sample impulse at each position."""
    pcm = np.zeros(n, dtype="<i2")
    for pos, amp in positions:
        pcm[pos] = amp
    return pcm


def post_onsets(pcm, min_relative_flux=0.1, channels=1, rate=RATE):
    payload = {"wav_base64": wav_b64(pcm, channels, rate)}
    if min_relative_flux is not None:
        payload["min_relative_flux"] = min_relative_flux
    return client.post("/v1/onsets", json=payload)


def post_onsets_raw(pcm, min_relative_flux=0.1):
    # Raw body so non-finite floats (NaN/Infinity, allowed by Python's json
    # encoder and decoder) actually reach the server.
    payload = {"wav_base64": wav_b64(pcm), "min_relative_flux": min_relative_flux}
    return client.post(
        "/v1/onsets",
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )


def test_impulse_train_detects_one_onset_per_click():
    pcm = clicks([(24000, STRONG), (48000, STRONG), (72000, STRONG)])
    resp = post_onsets(pcm)
    assert resp.status_code == 200
    onsets = resp.json()["onsets"]
    assert len(onsets) == 3
    indices = [o["frame_index"] for o in onsets]
    # One onset within half a window of each click, in ascending time order.
    for index, pos in zip(indices, (24000, 48000, 72000)):
        assert abs(index - pos) <= 512
        assert 0 <= index < N
    assert indices == sorted(indices)
    strengths = [o["strength"] for o in onsets]
    assert all(0.0 < s <= 1.0 for s in strengths)
    # The strongest onset is normalised to the whole-file maximum flux.
    assert max(strengths) == 1.0


def test_silence_returns_empty_onsets():
    resp = post_onsets(np.zeros(N, dtype="<i2"))
    assert resp.status_code == 200
    assert resp.json()["onsets"] == []


def test_default_min_relative_flux_is_0_1():
    pcm = clicks([(24000, WEAK), (48000, STRONG)])
    defaulted = post_onsets(pcm, min_relative_flux=None)
    explicit = post_onsets(pcm, min_relative_flux=0.1)
    assert defaulted.status_code == explicit.status_code == 200
    assert defaulted.json() == explicit.json()


def test_min_relative_flux_filters_weak_onsets():
    pcm = clicks([(24000, WEAK), (48000, STRONG)])
    both = post_onsets(pcm, min_relative_flux=0.1).json()["onsets"]
    assert len(both) == 2
    strong_only = post_onsets(pcm, min_relative_flux=0.5).json()["onsets"]
    assert len(strong_only) == 1
    assert abs(strong_only[0]["frame_index"] - 48000) <= 512


def test_onsets_kept_at_least_80ms_apart():
    # Two clicks 40 ms apart: only the stronger one survives.
    pcm = clicks([(48000, STRONG), (49920, STRONG // 2), (72000, STRONG)])
    onsets = post_onsets(pcm).json()["onsets"]
    assert len(onsets) == 2
    indices = [o["frame_index"] for o in onsets]
    assert abs(indices[0] - 48000) <= 512
    assert abs(indices[1] - 72000) <= 512
    assert indices[1] - indices[0] >= 3840  # 80 ms at 48 kHz


def test_equal_flux_ties_keep_earliest_candidate():
    # Two identical clicks 40 ms apart: the earlier one wins the tie.
    pcm = clicks([(48000, STRONG), (49920, STRONG)])
    onsets = post_onsets(pcm).json()["onsets"]
    assert len(onsets) == 1
    assert abs(onsets[0]["frame_index"] - 48000) <= 512


def test_stereo_detects_onsets_from_either_channel():
    left = clicks([(48000, STRONG)])
    right = clicks([(72000, STRONG)])
    pcm = np.stack([left, right], axis=1)
    resp = post_onsets(pcm, channels=2)
    assert resp.status_code == 200
    onsets = resp.json()["onsets"]
    assert len(onsets) == 2
    assert abs(onsets[0]["frame_index"] - 48000) <= 512
    assert abs(onsets[1]["frame_index"] - 72000) <= 512


def test_plateau_peak_takes_earliest_frame():
    flux = np.array([0.0, 1.0, 2.0, 2.0, 2.0, 1.0, 0.0])
    assert [i for i in range(len(flux)) if _is_local_peak(flux, i)] == [2]


def test_plateau_on_rising_slope_is_not_a_peak():
    flux = np.array([0.0, 2.0, 2.0, 3.0, 2.0])
    assert [i for i in range(len(flux)) if _is_local_peak(flux, i)] == [3]


@pytest.mark.parametrize("min_relative_flux", [-0.1, 1.1])
def test_min_relative_flux_out_of_range_returns_422(min_relative_flux):
    pcm = clicks([(48000, STRONG)])
    assert post_onsets(pcm, min_relative_flux).status_code == 422


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_min_relative_flux_returns_422(bad):
    pcm = clicks([(48000, STRONG)])
    assert post_onsets_raw(pcm, bad).status_code == 422


def test_invalid_base64_returns_422():
    resp = client.post(
        "/v1/onsets",
        json={"wav_base64": "!!!not-base64!!!", "min_relative_flux": 0.1},
    )
    assert resp.status_code == 422


def test_garbage_wav_returns_422():
    resp = client.post(
        "/v1/onsets",
        json={
            "wav_base64": base64.b64encode(b"not a wav").decode(),
            "min_relative_flux": 0.1,
        },
    )
    assert resp.status_code == 422


def test_non_compliant_wav_returns_422():
    pcm = clicks([(48000, STRONG)])
    assert post_onsets(pcm, rate=44100).status_code == 422
    # 0.5 s is shorter than the 1 s minimum.
    assert post_onsets(clicks([(12000, STRONG)], n=N // 4)).status_code == 422


def test_missing_wav_base64_returns_422():
    assert client.post("/v1/onsets", json={}).status_code == 422
