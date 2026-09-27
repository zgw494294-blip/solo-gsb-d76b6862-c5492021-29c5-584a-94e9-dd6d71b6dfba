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
MS = 48


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


def make_sine(n=N, freq=440.0, amp=0.5, seed=0):
    t = np.arange(n) / RATE
    sig = amp * np.sin(2 * np.pi * freq * t)
    if seed:
        rng = np.random.default_rng(seed)
        sig = 0.7 * sig + 0.3 * rng.standard_normal(n)
    return np.round(sig * 32768).astype("<i2")


def post_repair(pcm, gaps, channels=1, rate=RATE):
    return client.post(
        "/v1/repair-gaps",
        json={"wav_base64": wav_b64(pcm, channels, rate), "gaps": gaps},
    )


def zero(pcm, start, end):
    out = pcm.copy()
    out[start:end] = 0
    return out


def test_predicted_repair_fills_short_gap():
    pcm = make_sine()
    start, end = RATE, RATE + MS  # 1 ms gap
    resp = post_repair(zero(pcm, start, end), [[start, end]])
    assert resp.status_code == 200
    data = resp.json()
    assert data["repairs"] == [
        {"start": start, "end": end, "method": "predicted"}
    ]
    params, out = decode_wav(data["wav_base64"])
    assert params == (1, 2, RATE)
    assert out.size == N
    # Every frame outside the gap is byte-identical to the input.
    kept = np.concatenate([pcm[:start], pcm[end:]])
    got = np.concatenate([out[:start], out[end:]])
    assert np.array_equal(got, kept)
    # The fill stays small-error against the real signal and in range.
    fill = out[start:end].astype(np.float64) / 32768.0
    truth = pcm[start:end].astype(np.float64) / 32768.0
    assert np.max(np.abs(fill - truth)) < 0.01


@pytest.mark.parametrize("gap_ms", [1, 5, 10, 20])
def test_gap_lengths_between_1_and_20_ms(gap_ms):
    pcm = make_sine(freq=300.0)
    length = gap_ms * MS
    start = RATE
    resp = post_repair(zero(pcm, start, start + length), [[start, start + length]])
    assert resp.status_code == 200
    assert resp.json()["repairs"][0]["method"] == "predicted"


def test_empty_gaps_returns_input_unchanged():
    pcm = make_sine()
    resp = post_repair(pcm, [])
    assert resp.status_code == 200
    data = resp.json()
    assert data["repairs"] == []
    _, out = decode_wav(data["wav_base64"])
    assert np.array_equal(out, pcm)


def test_multiple_gaps_repaired_in_order():
    pcm = make_sine()
    # Exactly 40 ms edge guards, exactly 60 ms between gaps.
    gaps = [
        (40 * MS, 45 * MS),
        (105 * MS, 110 * MS),
        (RATE, RATE + 5 * MS),
        (N - 45 * MS, N - 40 * MS),
    ]
    damaged = pcm
    for start, end in gaps:
        damaged = zero(damaged, start, end)
    resp = post_repair(damaged, [list(g) for g in gaps])
    assert resp.status_code == 200
    repairs = resp.json()["repairs"]
    assert [(r["start"], r["end"]) for r in repairs] == list(gaps)
    assert all(r["method"] == "predicted" for r in repairs)
    _, out = decode_wav(resp.json()["wav_base64"])
    assert out.size == N
    # Non-gap samples untouched even with maximally tight spacing.
    mask = np.ones(N, dtype=bool)
    for start, end in gaps:
        mask[start:end] = False
    assert np.array_equal(out[mask], pcm[mask])


def test_stereo_repairs_each_channel_independently():
    n = N
    t = np.arange(n) / RATE
    left = np.round(0.4 * np.sin(2 * np.pi * 440 * t) * 32768).astype("<i2")
    right = np.round(0.4 * np.sin(2 * np.pi * 660 * t + 0.5) * 32768).astype("<i2")
    pcm = np.stack([left, right], axis=1)
    start, end = RATE, RATE + 2 * MS
    damaged = pcm.copy()
    damaged[start:end] = 0
    resp = post_repair(damaged, [[start, end]], channels=2)
    assert resp.status_code == 200
    assert resp.json()["repairs"][0]["method"] == "predicted"
    params, out = decode_wav(resp.json()["wav_base64"])
    assert params == (2, 2, RATE)
    out = out.reshape(-1, 2)
    assert np.array_equal(
        np.concatenate([out[:start], out[end:]], axis=0),
        np.concatenate([pcm[:start], pcm[end:]], axis=0),
    )
    truth = pcm.reshape(-1, 2).astype(np.float64) / 32768.0
    got = out.astype(np.float64) / 32768.0
    assert np.max(np.abs(got[start:end] - truth[start:end])) < 0.01


def test_full_scale_overshoot_falls_back_to_linear():
    # Ascending ramp to a +full-scale peak at the gap, descending after:
    # both LPC extrapolations run past the PCM full-scale range.
    start, end = RATE, RATE + MS
    x = np.zeros(N)
    x[start - 30 * MS : start] = -1.0 + 2.0 * np.arange(30 * MS) / (30 * MS)
    x[end : end + 30 * MS] = 1.0 - 2.0 * np.arange(30 * MS) / (30 * MS)
    pcm = np.round(np.clip(x, -1.0, 1.0) * 32767).astype("<i2")
    resp = post_repair(zero(pcm, start, end), [[start, end]])
    assert resp.status_code == 200
    assert resp.json()["repairs"][0]["method"] == "linear"
    _, out = decode_wav(resp.json()["wav_base64"])
    got = out[start:end].astype(np.float64) / 32768.0
    left_value = pcm[start - 1] / 32768.0
    right_value = pcm[end] / 32768.0
    positions = (np.arange(end - start) + 1.0) / (end - start + 1)
    expected = left_value + (right_value - left_value) * positions
    assert np.allclose(got, expected, atol=2 / 32768)
    # Boundary continuity: first/last filled samples neighbour the edges.
    assert abs(got[0] - left_value) < 2 / 32768
    assert abs(got[-1] - right_value) < 2 / 32768
    # Samples outside the gap remain exactly as input.
    assert np.array_equal(
        np.concatenate([out[:start], out[end:]]),
        np.concatenate([pcm[:start], pcm[end:]]),
    )


def test_mixed_channel_fallback_labels_gap_linear():
    # Left channel: benign sine (predicts fine). Right channel: full-scale
    # peak ramp (forces linear fallback on that channel only).
    start, end = RATE, RATE + MS
    t = np.arange(N) / RATE
    left = np.round(0.3 * np.sin(2 * np.pi * 440 * t) * 32768).astype("<i2")
    ramp = np.zeros(N)
    ramp[start - 30 * MS : start] = -1.0 + 2.0 * np.arange(30 * MS) / (
        30 * MS
    )
    ramp[end : end + 30 * MS] = 1.0 - 2.0 * np.arange(30 * MS) / (30 * MS)
    right = np.round(np.clip(ramp, -1.0, 1.0) * 32767).astype("<i2")
    pcm = np.stack([left, right], axis=1)
    damaged = pcm.copy()
    damaged[start:end] = 0
    resp = post_repair(damaged, [[start, end]], channels=2)
    assert resp.status_code == 200
    assert resp.json()["repairs"][0]["method"] == "linear"


@pytest.mark.parametrize(
    "gap",
    [
        [40 * MS, 40 * MS + MS - 1],       # 47 frames: shorter than 1 ms
        [RATE, RATE + 20 * MS + 1],        # 961 frames: longer than 20 ms
        [RATE, RATE],                      # empty
        [40 * MS - 1, 41 * MS],            # 39.98 ms from the head
        [N - 40 * MS, N - 40 * MS + MS],   # end only 39 ms from tail
        [-1, MS],                          # negative start
        [RATE, N + 1],                     # end out of bounds
        [N, N + MS],                       # start out of bounds
    ],
)
def test_gap_geometry_violations_return_422(gap):
    pcm = make_sine()
    start, end = gap
    if 0 <= start < N:
        end_clamped = min(max(end, start + 1), N)
        pcm = zero(pcm, start, end_clamped)
    resp = post_repair(pcm, [gap])
    assert resp.status_code == 422


def test_gaps_closer_than_60_ms_return_422():
    pcm = make_sine()
    gaps = [[RATE, RATE + MS], [RATE + MS + 60 * MS - 1, RATE + 62 * MS - 1]]
    for start, end in gaps:
        pcm = zero(pcm, start, end)
    assert post_repair(pcm, gaps).status_code == 422


def test_gaps_exactly_60_ms_apart_accepted():
    pcm = make_sine()
    gaps = [[RATE, RATE + MS], [RATE + MS + 60 * MS, RATE + 62 * MS]]
    for start, end in gaps:
        pcm = zero(pcm, start, end)
    resp = post_repair(pcm, gaps)
    assert resp.status_code == 200
    assert len(resp.json()["repairs"]) == 2


def test_gaps_must_be_strictly_ascending():
    pcm = make_sine()
    gaps = [[RATE + 10 * MS, RATE + 11 * MS], [RATE, RATE + MS]]
    assert post_repair(pcm, gaps).status_code == 422


def test_overlapping_gaps_return_422():
    pcm = make_sine()
    gaps = [[RATE, RATE + 5 * MS], [RATE + 4 * MS, RATE + 8 * MS]]
    assert post_repair(pcm, gaps).status_code == 422


def test_nonzero_gap_returns_422():
    pcm = make_sine()
    # Valid geometry, but the gap is not zeroed.
    resp = post_repair(pcm, [[RATE, RATE + MS]])
    assert resp.status_code == 422


def test_nonzero_on_one_stereo_channel_returns_422():
    t = np.arange(N) / RATE
    left = np.round(0.3 * np.sin(2 * np.pi * 440 * t) * 32768).astype("<i2")
    right = left.copy()
    pcm = np.stack([left, right], axis=1)
    pcm[RATE : RATE + MS] = 0
    pcm[RATE + 5, 1] = 1  # one non-zero sample on the right channel
    resp = post_repair(pcm, [[RATE, RATE + MS]], channels=2)
    assert resp.status_code == 422


def test_silent_left_context_returns_422():
    # Signal exists only in the first 600 ms; gap near 1 s has silent context.
    pcm = np.zeros(N, dtype="<i2")
    pcm[: 600 * MS] = make_sine(n=N)[: 600 * MS]
    resp = post_repair(pcm, [[RATE, RATE + MS]])
    assert resp.status_code == 422


def test_silent_right_context_returns_422():
    pcm = np.zeros(N, dtype="<i2")
    pcm[1400 * MS :] = make_sine(n=N)[1400 * MS :]
    resp = post_repair(pcm, [[RATE, RATE + MS]])
    assert resp.status_code == 422


def test_one_silent_channel_context_returns_422():
    t = np.arange(N) / RATE
    left = np.round(0.3 * np.sin(2 * np.pi * 440 * t) * 32768).astype("<i2")
    right = np.zeros(N, dtype="<i2")
    pcm = np.stack([left, right], axis=1)
    resp = post_repair(pcm, [[RATE, RATE + MS]], channels=2)
    assert resp.status_code == 422


def test_invalid_base64_returns_422():
    resp = client.post(
        "/v1/repair-gaps",
        json={"wav_base64": "!!!not-base64!!!", "gaps": [[RATE, RATE + MS]]},
    )
    assert resp.status_code == 422


def test_garbage_wav_returns_422():
    resp = client.post(
        "/v1/repair-gaps",
        json={
            "wav_base64": base64.b64encode(b"not a wav").decode(),
            "gaps": [[RATE, RATE + MS]],
        },
    )
    assert resp.status_code == 422


def test_non_compliant_wav_returns_422():
    pcm = make_sine()
    assert post_repair(pcm, [[RATE, RATE + MS]], rate=44100).status_code == 422


@pytest.mark.parametrize("field", ["wav_base64", "gaps"])
def test_missing_field_returns_422(field):
    payload = {
        "wav_base64": wav_b64(make_sine()),
        "gaps": [[RATE, RATE + MS]],
    }
    del payload[field]
    assert client.post("/v1/repair-gaps", json=payload).status_code == 422
