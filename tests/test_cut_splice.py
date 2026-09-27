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


def make_ramp(n=N, seed=7):
    # Deterministic non-constant content so splices are easy to verify.
    rng = np.random.default_rng(seed)
    return rng.integers(-20000, 20000, size=n).astype("<i2")


def post_cut(pcm, intervals, crossfade_ms=0, channels=1, rate=RATE, cues=None):
    payload = {
        "wav_base64": wav_b64(pcm, channels, rate),
        "remove_intervals": intervals,
        "crossfade_ms": crossfade_ms,
    }
    if cues is not None:
        payload["cue_intervals"] = cues
    return client.post("/v1/cut-splice", json=payload)


def test_hard_splice_removes_middle():
    pcm = make_ramp()
    resp = post_cut(pcm, [[RATE, RATE + 5000]], crossfade_ms=0)
    assert resp.status_code == 200
    data = resp.json()
    assert data["output_frames"] == N - 5000
    params, out = decode_wav(data["wav_base64"])
    assert params == (1, 2, RATE)
    assert np.array_equal(out, np.concatenate([pcm[:RATE], pcm[RATE + 5000 :]]))


def test_empty_intervals_returns_input_unchanged():
    pcm = make_ramp()
    resp = post_cut(pcm, [], crossfade_ms=5)
    assert resp.status_code == 200
    data = resp.json()
    assert data["output_frames"] == N
    _, out = decode_wav(data["wav_base64"])
    assert np.array_equal(out, pcm)


def test_delete_prefix_and_suffix():
    pcm = make_ramp()
    resp = post_cut(pcm, [[0, 1000], [N - 2000, N]], crossfade_ms=0)
    assert resp.status_code == 200
    data = resp.json()
    assert data["output_frames"] == N - 3000
    _, out = decode_wav(data["wav_base64"])
    assert np.array_equal(out, pcm[1000 : N - 2000])


def test_multiple_intervals_hard_splice():
    pcm = make_ramp()
    cuts = [[1000, 2000], [5000, 6000], [9000, 9500]]
    resp = post_cut(pcm, cuts, crossfade_ms=0)
    assert resp.status_code == 200
    data = resp.json()
    kept = np.concatenate(
        [pcm[:1000], pcm[2000:5000], pcm[6000:9000], pcm[9500:]]
    )
    assert data["output_frames"] == kept.size
    _, out = decode_wav(data["wav_base64"])
    assert np.array_equal(out, kept)


def test_crossfade_blends_overlap_linearly():
    pcm = np.empty(N, dtype="<i2")
    pcm[: RATE] = 1000
    pcm[RATE :] = 3000
    fade = 48  # crossfade_ms = 1
    resp = post_cut(pcm, [[RATE - 100, RATE + 100]], crossfade_ms=1)
    assert resp.status_code == 200
    data = resp.json()
    kept = (RATE - 100) + (RATE - 100)
    assert data["output_frames"] == kept - fade
    params, out = decode_wav(data["wav_base64"])
    assert params == (1, 2, RATE)
    head = RATE - 100 - fade
    assert np.all(out[:head] == 1000)
    assert np.all(out[head + fade :] == 3000)
    mixed = out[head : head + fade].astype(np.float64)
    weight = np.linspace(0.0, 1.0, fade)
    expected = 1000.0 * (1.0 - weight) + 3000.0 * weight
    # endpoints included: first frame is fully left, last fully right
    assert mixed[0] == 1000
    assert mixed[-1] == 3000
    assert np.allclose(mixed, expected, atol=1.0)


def test_crossfade_frame_count_multiple_seams():
    pcm = make_ramp()
    cuts = [[10000, 20000], [40000, 50000]]
    fade = 48 * 20
    resp = post_cut(pcm, cuts, crossfade_ms=20)
    assert resp.status_code == 200
    data = resp.json()
    kept = N - 10000 - 10000
    assert data["output_frames"] == kept - 2 * fade
    _, out = decode_wav(data["wav_base64"])
    assert out.size == kept - 2 * fade


def test_crossfade_stereo_fades_every_channel():
    left = np.empty(N, dtype="<i2")
    left[: RATE] = 1000
    left[RATE :] = 3000
    right = np.empty(N, dtype="<i2")
    right[: RATE] = -500
    right[RATE :] = -2500
    pcm = np.stack([left, right], axis=1)
    fade = 48 * 2
    resp = post_cut(pcm, [[RATE - 100, RATE + 100]], crossfade_ms=2, channels=2)
    assert resp.status_code == 200
    data = resp.json()
    params, out = decode_wav(data["wav_base64"])
    assert params == (2, 2, RATE)
    out = out.reshape(-1, 2)
    head = RATE - 100 - fade
    weight = np.linspace(0.0, 1.0, fade)
    for col, (a, b) in enumerate([(1000.0, 3000.0), (-500.0, -2500.0)]):
        assert np.all(out[:head, col] == int(a))
        assert np.all(out[head + fade :, col] == int(b))
        mixed = out[head : head + fade, col].astype(np.float64)
        expected = a * (1.0 - weight) + b * weight
        assert mixed[0] == a
        assert mixed[-1] == b
        assert np.allclose(mixed, expected, atol=1.0)


def test_output_may_be_shorter_than_one_second():
    pcm = make_ramp()
    resp = post_cut(pcm, [[1000, N - 1000]], crossfade_ms=0)
    assert resp.status_code == 200
    data = resp.json()
    assert data["output_frames"] == 2000  # ~42 ms
    _, out = decode_wav(data["wav_base64"])
    assert np.array_equal(out, np.concatenate([pcm[:1000], pcm[N - 1000 :]]))


@pytest.mark.parametrize(
    "intervals",
    [
        [[100, 500], [400, 900]],      # overlapping
        [[100, 500], [500, 900]],      # touching (no kept frame between)
        [[500, 900], [100, 200]],      # not ascending
        [[0, N + 1]],                  # end out of bounds
        [[-1, 100]],                   # negative start
        [[N, N + 10]],                 # start out of bounds
        [[100, 100]],                  # empty interval
        [[500, 100]],                  # start after end
    ],
)
def test_invalid_intervals_return_422(intervals):
    assert post_cut(make_ramp(), intervals).status_code == 422


def test_delete_everything_returns_422():
    assert post_cut(make_ramp(), [[0, N]]).status_code == 422


def test_first_segment_too_short_for_fade_returns_422():
    # Kept: [0, 30) and [5000, N); 30 < F = 48.
    assert post_cut(make_ramp(), [[30, 5000]], crossfade_ms=1).status_code == 422


def test_last_segment_too_short_for_fade_returns_422():
    assert (
        post_cut(make_ramp(), [[5000, N - 30]], crossfade_ms=1).status_code == 422
    )


def test_middle_segment_too_short_for_two_fades_returns_422():
    # Kept middle segment [2000, 2060) has 60 frames < 2F = 96.
    cuts = [[1000, 2000], [2060, 3000]]
    assert post_cut(make_ramp(), cuts, crossfade_ms=1).status_code == 422


def test_minimal_fade_capacity_accepted():
    pcm = make_ramp()
    fade = 48
    # First/last segments exactly F, middle exactly 2F.
    cuts = [[fade, 5000], [5000 + 2 * fade, N - fade]]
    resp = post_cut(pcm, cuts, crossfade_ms=1)
    assert resp.status_code == 200
    kept = fade + 2 * fade + fade
    assert resp.json()["output_frames"] == kept - 2 * fade


@pytest.mark.parametrize("crossfade_ms", [-1, 21])
def test_crossfade_ms_out_of_range_returns_422(crossfade_ms):
    resp = post_cut(make_ramp(), [[100, 200]], crossfade_ms=crossfade_ms)
    assert resp.status_code == 422


def test_invalid_base64_returns_422():
    resp = client.post(
        "/v1/cut-splice",
        json={
            "wav_base64": "!!!not-base64!!!",
            "remove_intervals": [[0, 100]],
            "crossfade_ms": 0,
        },
    )
    assert resp.status_code == 422


def test_garbage_wav_returns_422():
    resp = client.post(
        "/v1/cut-splice",
        json={
            "wav_base64": base64.b64encode(b"not a wav").decode(),
            "remove_intervals": [[0, 100]],
            "crossfade_ms": 0,
        },
    )
    assert resp.status_code == 422


def test_non_compliant_wav_returns_422():
    assert post_cut(make_ramp(), [[0, 100]], rate=44100).status_code == 422


@pytest.mark.parametrize("field", ["wav_base64", "remove_intervals", "crossfade_ms"])
def test_missing_field_returns_422(field):
    payload = {
        "wav_base64": wav_b64(make_ramp()),
        "remove_intervals": [[0, 100]],
        "crossfade_ms": 0,
    }
    del payload[field]
    assert client.post("/v1/cut-splice", json=payload).status_code == 422


def test_no_cue_intervals_returns_empty_mapped_cues():
    pcm = make_ramp()
    resp = post_cut(pcm, [[RATE, RATE + 5000]], crossfade_ms=0)
    assert resp.status_code == 200
    data = resp.json()
    assert data["mapped_cues"] == []
    # Original behaviour is unchanged.
    assert data["output_frames"] == N - 5000
    _, out = decode_wav(data["wav_base64"])
    assert np.array_equal(out, np.concatenate([pcm[:RATE], pcm[RATE + 5000 :]]))


def test_empty_cue_intervals_returns_empty_mapped_cues():
    resp = post_cut(make_ramp(), [[0, 100]], cues=[])
    assert resp.status_code == 200
    assert resp.json()["mapped_cues"] == []


def test_cues_hard_splice_split_and_shift():
    # Kept: [0, 10000), [20000, 40000), [50000, N); output starts
    # 0, 10000, 30000 (F = 0).
    cuts = [[10000, 20000], [40000, 50000]]
    cues = [
        {"id": "before", "start": 5000, "end": 9000},
        {"id": "spanning", "start": 5000, "end": 60000},
        {"id": "deleted", "start": 12000, "end": 18000},
        {"id": "after", "start": 60000, "end": 70000},
    ]
    resp = post_cut(make_ramp(), cuts, cues=cues)
    assert resp.status_code == 200
    mapped = resp.json()["mapped_cues"]
    assert [m["id"] for m in mapped] == ["before", "spanning", "deleted", "after"]

    assert mapped[0]["fragments"] == [
        {"source_interval": [5000, 9000], "output_interval": [5000, 9000]}
    ]
    assert mapped[1]["fragments"] == [
        {"source_interval": [5000, 10000], "output_interval": [5000, 10000]},
        {"source_interval": [20000, 40000], "output_interval": [10000, 30000]},
        {"source_interval": [50000, 60000], "output_interval": [30000, 40000]},
    ]
    # Fully deleted: no fragments survive.
    assert mapped[2]["fragments"] == []
    assert mapped[3]["fragments"] == [
        {"source_interval": [60000, 70000], "output_interval": [40000, 50000]}
    ]


def test_cues_keep_request_order_and_may_overlap():
    cuts = [[1000, 2000]]
    cues = [
        {"id": "z-late", "start": 5000, "end": 6000},
        {"id": "a-early", "start": 100, "end": 1500},
        {"id": "overlap", "start": 120, "end": 180},
    ]
    resp = post_cut(make_ramp(), cuts, cues=cues)
    assert resp.status_code == 200
    mapped = resp.json()["mapped_cues"]
    assert [m["id"] for m in mapped] == ["z-late", "a-early", "overlap"]
    assert mapped[0]["fragments"] == [
        {"source_interval": [5000, 6000], "output_interval": [4000, 5000]}
    ]
    # The part inside the deleted [1000, 2000) disappears.
    assert mapped[1]["fragments"] == [
        {"source_interval": [100, 1000], "output_interval": [100, 1000]},
    ]
    # Overlapping cues are mapped independently.
    assert mapped[2]["fragments"] == [
        {"source_interval": [120, 180], "output_interval": [120, 180]}
    ]


def test_cues_full_range_maps_to_every_kept_segment():
    cues = [{"id": "all", "start": 0, "end": N}]
    resp = post_cut(make_ramp(), [[1000, 2000]], cues=cues)
    assert resp.status_code == 200
    assert resp.json()["mapped_cues"][0]["fragments"] == [
        {"source_interval": [0, 1000], "output_interval": [0, 1000]},
        {"source_interval": [2000, N], "output_interval": [1000, N - 1000]},
    ]


def test_cues_crossfade_seam_maps_both_sides_to_overlapping_output():
    fade = 48  # crossfade_ms = 1
    cuts = [[RATE - 100, RATE + 100]]
    cues = [
        {"id": "across", "start": RATE - 200, "end": RATE + 300},
        {"id": "gone", "start": RATE - 100, "end": RATE + 100},
    ]
    resp = post_cut(make_ramp(), cuts, crossfade_ms=1, cues=cues)
    assert resp.status_code == 200
    data = resp.json()
    assert data["output_frames"] == (RATE - 100) * 2 - fade
    mapped = data["mapped_cues"]

    # Kept segments [0, RATE-100) and [RATE+100, N); the second starts at
    # output frame (RATE-100) - F. The fragments flanking the crossfade
    # zone map to output intervals overlapping by exactly F frames; both
    # are kept, not merged.
    left_out = RATE - 200
    right_shift = (RATE - 100 - fade) - (RATE + 100)
    assert mapped[0]["fragments"] == [
        {
            "source_interval": [RATE - 200, RATE - 100],
            "output_interval": [left_out, RATE - 100],
        },
        {
            "source_interval": [RATE + 100, RATE + 300],
            "output_interval": [
                RATE + 100 + right_shift,
                RATE + 300 + right_shift,
            ],
        },
    ]
    frag0, frag1 = mapped[0]["fragments"]
    overlap = frag0["output_interval"][1] - frag1["output_interval"][0]
    assert overlap == fade

    # A cue fully inside the deleted region leaves no fragments.
    assert mapped[1]["fragments"] == []


def test_cues_crossfade_two_seams_deduct_f_per_seam():
    fade = 48 * 2  # crossfade_ms = 2
    cuts = [[10000, 20000], [40000, 50000]]
    # Kept: [0, 10000), [20000, 40000), [50000, N); output starts
    # 0, 10000 - F, 30000 - 2F.
    cues = [
        {"id": "seg2", "start": 50000, "end": 50100},
        {"id": "left-tail", "start": 39900, "end": 40000},
        {"id": "right-head", "start": 50000, "end": 50000 + fade},
    ]
    resp = post_cut(make_ramp(), cuts, crossfade_ms=2, cues=cues)
    assert resp.status_code == 200
    mapped = resp.json()["mapped_cues"]

    seg2_start = 30000 - 2 * fade
    assert mapped[0]["fragments"] == [
        {
            "source_interval": [50000, 50100],
            "output_interval": [seg2_start, seg2_start + 100],
        }
    ]
    # The last F frames of the middle segment and the first F frames of
    # the last segment share the same output crossfade window.
    assert mapped[1]["fragments"] == [
        {
            "source_interval": [39900, 40000],
            "output_interval": [29900 - fade, 30000 - fade],
        }
    ]
    assert mapped[2]["fragments"] == [
        {
            "source_interval": [50000, 50000 + fade],
            "output_interval": [seg2_start, seg2_start + fade],
        }
    ]
    left_out = mapped[1]["fragments"][0]["output_interval"]
    right_out = mapped[2]["fragments"][0]["output_interval"]
    assert left_out[1] == right_out[0] + fade  # overlap of exactly F frames


def test_cues_no_deletion_identity_mapping():
    cues = [{"id": "x", "start": 100, "end": 5000}]
    resp = post_cut(make_ramp(), [], crossfade_ms=5, cues=cues)
    assert resp.status_code == 200
    assert resp.json()["mapped_cues"][0]["fragments"] == [
        {"source_interval": [100, 5000], "output_interval": [100, 5000]}
    ]


def test_duplicate_cue_id_returns_422():
    cues = [
        {"id": "dup", "start": 0, "end": 100},
        {"id": "dup", "start": 200, "end": 300},
    ]
    assert post_cut(make_ramp(), [[0, 100]], cues=cues).status_code == 422


def test_empty_cue_id_returns_422():
    cues = [{"id": "", "start": 0, "end": 100}]
    assert post_cut(make_ramp(), [[0, 100]], cues=cues).status_code == 422


@pytest.mark.parametrize(
    "cue",
    [
        {"id": "empty", "start": 100, "end": 100},      # empty interval
        {"id": "reversed", "start": 500, "end": 100},   # start after end
        {"id": "negative", "start": -1, "end": 100},    # negative start
        {"id": "past-end", "start": 0, "end": N + 1},   # end out of bounds
        {"id": "beyond", "start": N, "end": N + 10},    # start out of bounds
    ],
)
def test_invalid_cue_interval_returns_422(cue):
    assert post_cut(make_ramp(), [[0, 100]], cues=[cue]).status_code == 422


def test_cue_count_limit():
    ok = [{"id": str(i), "start": i, "end": i + 1} for i in range(2000)]
    assert post_cut(make_ramp(), [[0, 1]], cues=ok).status_code == 200
    too_many = [{"id": str(i), "start": i, "end": i + 1} for i in range(2001)]
    assert post_cut(make_ramp(), [[0, 1]], cues=too_many).status_code == 422
