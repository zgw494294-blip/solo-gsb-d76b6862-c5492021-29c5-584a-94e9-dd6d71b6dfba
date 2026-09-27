import base64
import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.loudness import _quantize
from app.time_stretch import (
    HOP_FRAMES,
    SEARCH_FRAMES,
    SEGMENT_FRAMES,
    _choose_start,
    _place_segments,
    time_stretch_wav,
)

client = TestClient(app)

RATE = 48000


def wav_b64(pcm: np.ndarray, channels: int = 1, rate: int = RATE) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.reshape(-1, channels).astype("<i2").tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def decode_wav(b64: str):
    raw = base64.b64decode(b64)
    with wave.open(io.BytesIO(raw), "rb") as wf:
        params = (wf.getnchannels(), wf.getsampwidth(), wf.getframerate())
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
    return params, pcm


def make_tone(seconds=3.0, freq=440.0, seed=0, channels=1, amplitude=0.4):
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    rng = np.random.default_rng(seed)
    mono = (
        amplitude * np.sin(2 * np.pi * freq * t)
        + 0.1 * amplitude * np.sin(2 * np.pi * 2 * freq * t)
        + 0.005 * rng.standard_normal(n)
    )
    if channels == 1:
        return np.round(mono * 32768).astype("<i2")
    right = 0.7 * np.sin(2 * np.pi * (freq * 0.75) * t)
    return np.round(np.stack([mono, right], axis=1) * 32768).astype("<i2")


def post_stretch(pcm, speed, channels=1, rate=RATE):
    return client.post(
        "/v1/time-stretch",
        json={"wav_base64": wav_b64(pcm, channels, rate), "speed": speed},
    )


def dominant_freq(pcm: np.ndarray, rate: int = RATE) -> float:
    x = pcm.astype(np.float64)
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    freqs = np.fft.rfftfreq(x.size, 1.0 / rate)
    mask = freqs > 100.0
    return float(freqs[mask][np.argmax(spectrum[mask])])


@pytest.mark.parametrize("speed", [0.8, 0.9, 0.95, 1.05, 1.1, 1.2, 1.25])
def test_output_frame_count_is_round_input_over_speed(speed):
    pcm = make_tone()
    resp = post_stretch(pcm, speed)
    assert resp.status_code == 200
    data = resp.json()
    expected = round(pcm.shape[0] / speed)
    assert data["output_frames"] == expected
    params, out = decode_wav(data["wav_base64"])
    assert params == (1, 2, RATE)
    assert out.size == expected


@pytest.mark.parametrize("channels", [1, 2])
def test_keeps_sample_rate_bit_depth_and_channels(channels):
    pcm = make_tone(channels=channels)
    resp = post_stretch(pcm, 0.8, channels=channels)
    assert resp.status_code == 200
    params, out = decode_wav(resp.json()["wav_base64"])
    assert params == (channels, 2, RATE)
    assert out.size == round(pcm.shape[0] / 0.8) * channels
    assert out.reshape(-1, channels).shape[0] == round(pcm.shape[0] / 0.8)


def test_speed_one_leaves_pcm_unchanged():
    pcm = make_tone(seed=3)
    resp = post_stretch(pcm, 1.0)
    assert resp.status_code == 200
    data = resp.json()
    assert data["output_frames"] == pcm.shape[0]
    _, out = decode_wav(data["wav_base64"])
    assert np.array_equal(out, pcm.reshape(-1))


def test_first_segment_is_exact_input_head():
    # The first piece owns output [0, HOP) before any fade, so those frames
    # are the input head exactly.
    pcm = make_tone(seed=5)
    resp = post_stretch(pcm, 0.8)
    _, out = decode_wav(resp.json()["wav_base64"])
    assert np.array_equal(out[:HOP_FRAMES], pcm[:HOP_FRAMES])


def test_last_segment_is_exact_input_tail():
    # The tail piece's faded head spans the full overlap with the previous
    # piece (SEGMENT_FRAMES - final_hop frames when the final hop is
    # shortened); after it, the remaining PCM is the input tail copied
    # sample-for-sample.
    pcm = make_tone(seed=5)
    speed = 0.8
    resp = post_stretch(pcm, speed)
    data = resp.json()
    _, out = decode_wav(data["wav_base64"])
    tail_pos = data["output_frames"] - SEGMENT_FRAMES
    final_hop = tail_pos % HOP_FRAMES or HOP_FRAMES
    overlap = SEGMENT_FRAMES - final_hop
    assert np.array_equal(
        out[tail_pos + overlap :],
        pcm[pcm.shape[0] - SEGMENT_FRAMES + overlap :],
    )


@pytest.mark.parametrize("channels", [1, 2])
def test_shortened_final_hop_fades_last_piece_over_full_overlap(channels):
    # Regression: 144000 frames @ speed 0.8 -> 180000 output frames; the
    # tail piece sits at 178080, so the final hop is shortened to 480 and
    # the true overlap with the previous segment is 1920 - 480 = 1440
    # frames. The whole zone must fade the already-mixed output into the
    # last piece with a linear 0 -> 1 ramp (one weight sequence shared by
    # all channels), then copy the last piece's remaining PCM verbatim.
    pcm = make_tone(seconds=3.0, seed=7, channels=channels)
    pcm2d = pcm.reshape(-1, channels)
    samples = pcm2d.astype(np.float64) / 32768.0
    output_frames = round(pcm2d.shape[0] / 0.8)
    starts, placements = _place_segments(samples, output_frames)
    tail_pos, final_hop = placements[-1]
    assert (tail_pos, final_hop) == (output_frames - SEGMENT_FRAMES, 480)
    overlap = SEGMENT_FRAMES - final_hop
    assert overlap == 1440

    resp = post_stretch(pcm, 0.8, channels=channels)
    assert resp.status_code == 200
    _, out = decode_wav(resp.json()["wav_base64"])
    out = out.reshape(-1, channels)

    # Pre-existing mix in the overlap zone: the previous seam's blend up to
    # its end, then the previous piece's raw tail.
    prev_pos = placements[-2][0]
    prev_start, prev_prev_start = starts[-2], starts[-3]
    fade_in = np.linspace(0.0, 1.0, HOP_FRAMES).reshape(-1, 1)
    seam = (
        samples[prev_prev_start + HOP_FRAMES : prev_prev_start + SEGMENT_FRAMES]
        * (1.0 - fade_in)
        + samples[prev_start : prev_start + HOP_FRAMES] * fade_in
    )
    seam_offset = tail_pos - prev_pos
    seam_remaining = HOP_FRAMES - seam_offset
    existing = np.empty((overlap, channels))
    existing[:seam_remaining] = seam[seam_offset:]
    existing[seam_remaining:] = samples[
        prev_start + HOP_FRAMES : prev_start + SEGMENT_FRAMES
    ]

    weights = np.linspace(0.0, 1.0, overlap).reshape(-1, 1)
    expected = (
        existing * (1.0 - weights)
        + samples[starts[-1] : starts[-1] + overlap] * weights
    )
    assert np.array_equal(
        out[tail_pos : tail_pos + overlap], _quantize(expected)
    )
    # The ramp is not restarted or cut short: it begins at the existing mix
    # and ends exactly on the last piece, whose remaining PCM (the input
    # tail) is then copied sample-for-sample.
    assert np.array_equal(out[tail_pos], _quantize(existing[:1])[0])
    assert np.array_equal(
        out[tail_pos + overlap - 1],
        pcm2d[pcm2d.shape[0] - SEGMENT_FRAMES + overlap - 1],
    )
    assert np.array_equal(
        out[tail_pos + overlap :],
        pcm2d[pcm2d.shape[0] - SEGMENT_FRAMES + overlap :],
    )


def test_full_length_final_hop_keeps_regular_overlap():
    # 96000 frames @ speed 1.25 -> 76800 output frames; the tail piece sits
    # on a regular hop boundary (74880 = 78 * 960), so the final hop is a
    # full 960 and the last piece fades over the regular 960-frame overlap.
    pcm = make_tone(seconds=2.0, seed=9)
    output_frames = round(pcm.shape[0] / 1.25)
    assert output_frames % HOP_FRAMES == 0
    samples = pcm.reshape(-1, 1).astype(np.float64) / 32768.0
    _, placements = _place_segments(samples, output_frames)
    tail_pos, final_hop = placements[-1]
    assert final_hop == HOP_FRAMES
    resp = post_stretch(pcm, 1.25)
    assert resp.status_code == 200
    _, out = decode_wav(resp.json()["wav_base64"])
    # The tail piece's exclusive tail starts one regular overlap in.
    assert np.array_equal(
        out[tail_pos + HOP_FRAMES :],
        pcm[pcm.shape[0] - SEGMENT_FRAMES + HOP_FRAMES :],
    )


def test_shortened_final_hop_hits_exact_frame_count():
    # 144000 / 0.8 = 180000; tail piece starts at 178080, i.e. the final hop
    # is shortened from 960 to 480 frames.
    pcm = make_tone(seconds=3.0)
    resp = post_stretch(pcm, 0.8)
    data = resp.json()
    assert data["output_frames"] == 180000
    _, out = decode_wav(data["wav_base64"])
    assert out.size == 180000


def test_pitch_is_preserved_while_duration_changes():
    pcm = make_tone(freq=440.0, seed=2)
    input_freq = dominant_freq(pcm)
    assert input_freq == pytest.approx(440.0, abs=1.0)
    for speed in (0.8, 1.25):
        resp = post_stretch(pcm, speed)
        _, out = decode_wav(resp.json()["wav_base64"])
        assert dominant_freq(out) == pytest.approx(440.0, abs=2.0)


def test_stereo_uses_one_joint_start_for_all_channels():
    # The common start is selected by the correlation summed over channels:
    # making one channel ambiguous (a constant tone) while the other carries
    # a distinctive noise waveform must still follow the distinctive channel.
    n = 3 * RATE
    rng = np.random.default_rng(11)
    t = np.arange(n) / RATE
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    noise = 0.3 * rng.standard_normal(n)
    stereo = np.stack([tone, noise], axis=1)
    mono_noise = noise[:, np.newaxis]

    previous_start = 48_000
    nominal = previous_start + HOP_FRAMES + 131
    joint = _choose_start(stereo, nominal, previous_start)
    noise_only = _choose_start(mono_noise, nominal, previous_start)
    assert joint == noise_only
    assert abs(joint - nominal) <= SEARCH_FRAMES


def test_stretch_keeps_stereo_framewise_channel_alignment():
    # Both channels of every output frame come from the same input frame:
    # the inter-channel lag of the input (right is left rolled by 7 frames)
    # must survive stretching unchanged.
    n = 3 * RATE
    rng = np.random.default_rng(4)
    left = rng.standard_normal(n)
    shift = 7
    right = np.roll(left, shift) + 0.05 * rng.standard_normal(n)
    pcm = np.round(
        np.stack([0.3 * left, 0.3 * right], axis=1) * 32768
    ).astype("<i2")
    resp = post_stretch(pcm, 0.8, channels=2)
    assert resp.status_code == 200
    _, out = decode_wav(resp.json()["wav_base64"])
    out = out.reshape(-1, 2).astype(np.float64)
    l, r = out[:, 0], out[:, 1]
    l -= l.mean()
    r -= r.mean()
    l_in, r_in = pcm[:, 0].astype(np.float64), pcm[:, 1].astype(np.float64)
    l_in -= l_in.mean()
    r_in -= r_in.mean()
    lags = np.arange(-32, 33)

    def peak_lag(a, b):
        a = a - a.mean()
        b = b - b.mean()
        corr = np.array([np.sum(a * np.roll(b, lag)) for lag in lags])
        return int(lags[np.argmax(corr)])

    assert peak_lag(l_in, r_in) == -shift
    assert peak_lag(l, r) == -shift


def test_choose_start_picks_period_match_near_nominal():
    # Sine with period 96 frames: shifting the reference by two periods the
    # candidate 2*96 away must win, but the search is clipped to +/-240...
    # build a signal whose best match inside the window sits at nominal+96.
    n = 4 * RATE
    t = np.arange(n) / RATE
    samples = (0.5 * np.sin(2 * np.pi * 440 * t))[:, np.newaxis]
    previous_start = 10_000
    nominal = previous_start + HOP_FRAMES + 96
    chosen = _choose_start(samples, nominal, previous_start)
    # The chosen start must correlate best; for a pure tone the periodic
    # match nearest the nominal wins, and it stays within +/-240 frames.
    assert abs(chosen - nominal) <= SEARCH_FRAMES
    assert chosen >= 0 and chosen + SEGMENT_FRAMES <= n


def test_choose_start_zero_reference_returns_nearest_position():
    n = 4 * RATE
    rng = np.random.default_rng(1)
    samples = rng.standard_normal((n, 1)) * 0.1
    # Place the previous piece so its overlap reference is all zeros.
    previous_start = 5000
    samples[
        previous_start
        + SEGMENT_FRAMES
        - HOP_FRAMES : previous_start
        + SEGMENT_FRAMES
    ] = 0.0
    nominal = previous_start + HOP_FRAMES + 37
    chosen = _choose_start(samples, nominal, previous_start)
    assert chosen == nominal


def test_choose_start_zero_candidates_returns_nearest_position():
    n = 4 * RATE
    samples = np.zeros((n, 1), dtype=np.float64)
    # Energize only the reference overlap block; put the nominal position
    # far enough away that every candidate head window within +/-240 misses
    # that block, so all candidates have zero energy.
    previous_start = 20_000
    ref_start = previous_start + SEGMENT_FRAMES - HOP_FRAMES
    samples[ref_start : ref_start + HOP_FRAMES] = 1.0
    nominal = ref_start + 3000
    chosen = _choose_start(samples, nominal, previous_start)
    assert chosen == nominal


def test_choose_start_ties_prefer_nearest_then_smaller_index():
    # Constant non-zero signal: every candidate NCC equals exactly 1.0, so
    # the tie rule (nearest nominal, then smaller index) must apply.
    n = 4 * RATE
    samples = np.full((n, 1), 0.25, dtype=np.float64)
    previous_start = 30_000
    nominal = previous_start + HOP_FRAMES + 10
    chosen = _choose_start(samples, nominal, previous_start)
    assert chosen == nominal

    # Equal-distance exact ties with a periodic waveform: phase-flipping the
    # reference by half a period makes only the +/- half-period candidates
    # match perfectly (NCC = 1), and the smaller index must win.
    period = 200
    half = period // 2
    samples3 = np.zeros((n, 1), dtype=np.float64)
    previous_start3 = 50_000
    ref_start3 = previous_start3 + SEGMENT_FRAMES - HOP_FRAMES
    idx = np.arange(n)
    periodic = np.where((idx % period) < half, 1.0, -1.0)
    # Reference region stores the half-period-shifted phase; candidate
    # region far away stores the normal periodic phase.
    samples3[:, 0] = periodic
    samples3[ref_start3 : ref_start3 + HOP_FRAMES, 0] = periodic[
        ref_start3 - half : ref_start3 - half + HOP_FRAMES
    ]
    nominal3 = ref_start3 + 4000
    chosen3 = _choose_start(samples3, nominal3, previous_start3)
    assert chosen3 == nominal3 - half  # exact tie -> smaller index
    assert abs(chosen3 - nominal3) == half <= SEARCH_FRAMES


@pytest.mark.parametrize("speed", [0.79, 1.26, -1.0, 0.0])
def test_speed_out_of_range_returns_422(speed):
    resp = post_stretch(make_tone(), speed)
    assert resp.status_code == 422


def test_missing_field_returns_422():
    resp = client.post(
        "/v1/time-stretch", json={"wav_base64": wav_b64(make_tone())}
    )
    assert resp.status_code == 422


def test_invalid_base64_returns_422():
    resp = client.post(
        "/v1/time-stretch",
        json={"wav_base64": "!!!not-base64!!!", "speed": 1.0},
    )
    assert resp.status_code == 422


def test_garbage_wav_returns_422():
    resp = client.post(
        "/v1/time-stretch",
        json={
            "wav_base64": base64.b64encode(b"not a wav").decode(),
            "speed": 1.0,
        },
    )
    assert resp.status_code == 422


def test_non_compliant_wav_returns_422():
    assert post_stretch(make_tone(), 1.0, rate=44100).status_code == 422


def test_core_validates_speed_range():
    pcm = make_tone()
    raw = base64.b64decode(wav_b64(pcm))
    with pytest.raises(Exception):
        time_stretch_wav(raw, 1.3)
