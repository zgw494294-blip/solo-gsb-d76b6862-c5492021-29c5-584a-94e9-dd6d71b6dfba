import base64
import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

RATE = 48000
WINDOW = RATE // 10  # 100 ms = 4800 frames
MAX_SHIFT = WINDOW   # +/-100 ms


def wav_b64(pcm: np.ndarray, channels: int = 1, rate: int = RATE) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.astype("<i2").tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def pcm_from_float(sig: np.ndarray) -> np.ndarray:
    return np.clip(np.round(sig * 32768.0), -32768, 32767).astype("<i2")


def make_tone(duration=2.0, freq=997.0, amp=0.2, channels=1, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(RATE * duration)) / RATE
    # Slight FM + noise keeps the signal non-periodic so the correlation
    # peak at the true shift is unique.
    phase = 2 * np.pi * freq * t + 0.3 * np.sin(2 * np.pi * 3.1 * t)
    sig = amp * np.sin(phase) + 0.01 * rng.standard_normal(t.size)
    pcm = pcm_from_float(sig)
    if channels == 2:
        pcm = np.stack([pcm, np.roll(pcm, 137)], axis=1)
    return pcm


def shifted_gain(pcm, shift, gain):
    """cand[i + shift] = gain * ref[i] (shift > 0 delays the candidate)."""
    n_ref = pcm.shape[0]
    if pcm.ndim == 1:
        cand = np.zeros(n_ref, dtype=pcm.dtype)
    else:
        cand = np.zeros((n_ref, pcm.shape[1]), dtype=pcm.dtype)
    ref_idx = np.arange(max(-shift, 0), min(n_ref, n_ref - shift))
    cand_idx = ref_idx + shift
    cand[cand_idx] = np.clip(
        np.round(pcm[ref_idx].astype(np.float64) * gain), -32768, 32767
    ).astype("<i2")
    return cand


def channels_of(pcm):
    return 1 if pcm.ndim == 1 else pcm.shape[1]


def post_compare(ref_pcm, cand_pcm):
    return client.post(
        "/v1/compare-gain",
        json={
            "reference_wav_base64": wav_b64(ref_pcm, channels_of(ref_pcm)),
            "candidate_wav_base64": wav_b64(cand_pcm, channels_of(cand_pcm)),
        },
    )


def test_identical_files():
    ref = make_tone()
    resp = post_compare(ref, ref)
    assert resp.status_code == 200
    data = resp.json()
    assert data["offset_samples"] == 0
    assert data["gain_db"] == pytest.approx(0.0, abs=0.001)
    assert data["polarity_inverted"] is False
    assert data["reference_unmatched_samples"] == 0
    assert data["candidate_unmatched_samples"] == 0
    assert data["anomaly_intervals"] == []
    assert data["overlap_gain_only"] is True


def test_gain_only_no_shift():
    ref = make_tone(amp=0.15)
    cand = pcm_from_float(ref.astype(np.float64) / 32768.0 * 2.0)
    resp = post_compare(ref, cand)
    assert resp.status_code == 200
    data = resp.json()
    assert data["offset_samples"] == 0
    assert data["gain_db"] == pytest.approx(6.0206, abs=0.01)
    assert data["polarity_inverted"] is False
    assert data["overlap_gain_only"] is True


def test_positive_shift_with_gain():
    ref = make_tone(duration=2.0)
    shift = 1234
    cand = shifted_gain(ref, shift, 1.0)
    resp = post_compare(ref, cand)
    assert resp.status_code == 200
    data = resp.json()
    assert data["offset_samples"] == shift
    assert data["gain_db"] == pytest.approx(0.0, abs=0.001)
    assert data["reference_unmatched_samples"] == shift
    assert data["candidate_unmatched_samples"] == shift


def test_negative_shift():
    ref = make_tone(duration=2.0, seed=3)
    shift = -2000
    cand = shifted_gain(ref, shift, 1.0)
    resp = post_compare(ref, cand)
    assert resp.status_code == 200
    data = resp.json()
    assert data["offset_samples"] == shift
    assert data["reference_unmatched_samples"] == -shift
    assert data["candidate_unmatched_samples"] == -shift


def test_polarity_inversion():
    ref = make_tone(duration=2.0)
    shift = 500
    cand = shifted_gain(ref, shift, -1.0)
    resp = post_compare(ref, cand)
    assert resp.status_code == 200
    data = resp.json()
    assert data["offset_samples"] == shift
    assert data["gain_db"] == pytest.approx(0.0, abs=0.001)
    assert data["polarity_inverted"] is True
    assert data["overlap_gain_only"] is True


def test_stereo_shared_gain_and_shift():
    ref = make_tone(duration=2.0, channels=2)
    shift = -800
    cand = shifted_gain(ref, shift, 0.5)
    resp = post_compare(ref, cand)
    assert resp.status_code == 200
    data = resp.json()
    assert data["offset_samples"] == shift
    assert data["gain_db"] == pytest.approx(-6.0206, abs=0.02)
    assert data["polarity_inverted"] is False
    assert data["overlap_gain_only"] is True


def test_different_lengths_unmatched_counts():
    # 2 s reference; candidate is a 1.5 s extract delayed so that candidate
    # frame (i + shift) equals reference frame i:
    #   cand[shift:] = ref[0 : n_cand - shift], cand[:shift] = 0
    ref = make_tone(duration=2.0, seed=7)
    shift = 1440
    n_cand = RATE * 3 // 2
    cand = np.zeros(n_cand, dtype=ref.dtype)
    cand[shift:] = ref[: n_cand - shift]
    resp = post_compare(ref, cand)
    assert resp.status_code == 200
    data = resp.json()
    assert data["offset_samples"] == shift
    # Overlap spans ref frames [0, 70560) and cand frames [1440, 72000):
    # reference has 25440 trailing frames uncovered; the candidate has 1440
    # leading frames outside the overlap.
    assert data["reference_unmatched_samples"] == 2 * RATE - (n_cand - shift)
    assert data["candidate_unmatched_samples"] == shift
    assert data["overlap_gain_only"] is True


def _add_burst(pcm, start, end, amp=0.05, freq=8000.0):
    """Add an orthogonal high-frequency burst to selected frames."""
    out = pcm.astype(np.float64) / 32768.0
    t = np.arange(start, end) / RATE
    out[start:end] += amp * np.sin(2 * np.pi * freq * t)
    return pcm_from_float(out)


def test_local_anomaly_flagged_and_merged():
    ref = make_tone(duration=2.0, amp=0.2, seed=11)
    cand = _add_burst(ref, 2 * WINDOW, 3 * WINDOW)
    resp = post_compare(ref, cand)
    assert resp.status_code == 200
    data = resp.json()
    assert data["offset_samples"] == 0
    assert data["overlap_gain_only"] is False
    assert data["anomaly_intervals"] == [[200.0, 300.0]]


def test_two_separate_anomaly_windows_stay_separate():
    ref = make_tone(duration=2.0, amp=0.2, seed=13)
    cand = ref.copy()
    for w in (1, 5):
        cand = _add_burst(cand, w * WINDOW, (w + 1) * WINDOW)
    resp = post_compare(ref, cand)
    data = resp.json()
    assert data["overlap_gain_only"] is False
    assert data["anomaly_intervals"] == [[100.0, 200.0], [500.0, 600.0]]


def test_adjacent_anomaly_windows_merge():
    ref = make_tone(duration=2.0, amp=0.2, seed=17)
    cand = _add_burst(ref, 3 * WINDOW, 6 * WINDOW)
    resp = post_compare(ref, cand)
    data = resp.json()
    assert data["overlap_gain_only"] is False
    assert data["anomaly_intervals"] == [[300.0, 600.0]]


def test_tail_window_included():
    # 1.5 s file: the last window is 500 ms long and still scored.
    n = RATE * 3 // 2
    ref = make_tone(duration=1.5, amp=0.2, seed=19)
    lo = RATE + WINDOW // 2  # 1.25 s, inside the 500 ms tail window
    cand = _add_burst(ref, lo, n)
    resp = post_compare(ref, cand)
    data = resp.json()
    assert data["overlap_gain_only"] is False
    assert data["anomaly_intervals"] == [[1000.0, 1500.0]]


def test_shift_boundary_plus_minus_100ms():
    ref = make_tone(duration=2.0, seed=23)
    for shift in (MAX_SHIFT, -MAX_SHIFT):
        cand = shifted_gain(ref, shift, 1.0)
        resp = post_compare(ref, cand)
        assert resp.status_code == 200
        assert resp.json()["offset_samples"] == shift


def test_shift_beyond_100ms_not_followed():
    # Candidate delayed 200 ms: the search must not reach it; the best fit
    # inside +/-100 ms is at the boundary and residuals fail the gain test.
    ref = make_tone(duration=2.0, amp=0.2, seed=29)
    shift = 2 * MAX_SHIFT
    cand = shifted_gain(ref, shift, 1.0)
    resp = post_compare(ref, cand)
    data = resp.json()
    assert data["offset_samples"] != shift
    assert data["overlap_gain_only"] is False
    assert data["anomaly_intervals"]


def test_silence_returns_422():
    silent = np.zeros(2 * RATE, dtype="<i2")
    resp = post_compare(silent, make_tone())
    assert resp.status_code == 422
    resp = post_compare(make_tone(), silent)
    assert resp.status_code == 422


def test_channel_mismatch_returns_422():
    mono = make_tone(channels=1)
    stereo = make_tone(channels=2)
    resp = post_compare(mono, stereo)
    assert resp.status_code == 422
    assert "channel" in resp.json()["detail"].lower()


def test_invalid_base64_returns_422():
    resp = client.post(
        "/v1/compare-gain",
        json={
            "reference_wav_base64": "!!!not-base64!!!",
            "candidate_wav_base64": wav_b64(make_tone()),
        },
    )
    assert resp.status_code == 422


def test_garbage_wav_returns_422():
    resp = client.post(
        "/v1/compare-gain",
        json={
            "reference_wav_base64": base64.b64encode(b"not a wav").decode(),
            "candidate_wav_base64": wav_b64(make_tone()),
        },
    )
    assert resp.status_code == 422


def test_non_compliant_wav_returns_422():
    # 44.1 kHz candidate
    bad = make_tone()
    resp = client.post(
        "/v1/compare-gain",
        json={
            "reference_wav_base64": wav_b64(make_tone()),
            "candidate_wav_base64": wav_b64(bad, rate=44100),
        },
    )
    assert resp.status_code == 422


def test_missing_field_returns_422():
    resp = client.post(
        "/v1/compare-gain",
        json={"reference_wav_base64": wav_b64(make_tone())},
    )
    assert resp.status_code == 422


def test_periodic_signal_identical_tie_picks_zero_offset():
    # Pure period with identical content: every period-multiple lag
    # (including 0 and +/-200) correlates at exactly 1; the smallest
    # absolute lag, 0, must win.
    t = np.arange(2 * RATE) / RATE
    pcm = pcm_from_float(0.2 * np.sin(2 * np.pi * 1000 * t))
    resp = post_compare(pcm, pcm)
    assert resp.status_code == 200
    assert resp.json()["offset_samples"] == 0


def test_periodic_signal_symmetric_tie_picks_smallest_lag():
    # Alternating A,B periods (period 2p): lags +/-p correlate at -1 while
    # lag 0 and +/-2p correlate at +1 -- every period-multiple lag ties on
    # |corr| = 1, so the smallest absolute lag (0) wins. Then a candidate
    # shifted by one period makes +/-p the inner +/- pair with +1 and 0
    # anti-correlate: the numerically smallest lag (-p) wins.
    rng = np.random.default_rng(31)
    pa = rng.standard_normal(48)
    pb = rng.standard_normal(48)
    pair = np.concatenate([pa, pb])
    ref = pcm_from_float(0.2 * np.tile(pair, 1000) / np.max(np.abs(pair)))
    p = 48
    cand = np.empty_like(ref)
    cand[:-p] = ref[p:]      # cand[i+p] = ref[i]  -> offset +p
    cand[-p:] = ref[:p]
    resp = post_compare(ref, cand)
    assert resp.status_code == 200
    data = resp.json()
    # +p and -p both line up perfectly for this periodic pattern; lag 0 is
    # inverted. Smallest |lag| among +/-p is a tie, then smallest numeric.
    assert data["offset_samples"] == -p
    assert data["polarity_inverted"] is False


def test_near_periodic_rotation_picks_true_peak_over_zero_lag():
    # 1 s mono: 10000 LSB sine with a 100-sample period plus a single
    # sample bumped by 500 LSB; the candidate is the reference rotated
    # right by exactly one period. Lag 0 and lag 100 land in the same FFT
    # error band, but only lag 100 aligns the overlap exactly (|corr| = 1),
    # so the exact re-check must pick 100 over the rounded tie at 0.
    t = np.arange(RATE) / RATE
    pcm = np.round(10000.0 * np.sin(2 * np.pi * (RATE / 100) * t)).astype("<i2")
    pcm[12345] += 500
    cand = np.roll(pcm, 100)
    resp = post_compare(pcm, cand)
    assert resp.status_code == 200
    data = resp.json()
    assert data["offset_samples"] == 100
    assert data["reference_unmatched_samples"] == 100
    assert data["candidate_unmatched_samples"] == 100
    assert data["gain_db"] == pytest.approx(0.0, abs=0.001)
    assert data["polarity_inverted"] is False
    assert data["anomaly_intervals"] == []
    assert data["overlap_gain_only"] is True


def test_near_periodic_rotation_long_stereo():
    # 10 s stereo periodic input: the near-tied FFT peaks (one per period
    # across +/-100 ms) must be re-checked lazily -- not with a full-length
    # dot product per peak -- and still resolve to the true rotation lag.
    n = 10 * RATE
    t = np.arange(n) / RATE
    sig = np.round(10000.0 * np.sin(2 * np.pi * (RATE / 100) * t)).astype("<i2")
    sig[123456] += 500
    stereo = np.stack([sig, np.roll(sig, 37)], axis=1)
    cand = np.roll(stereo, 100, axis=0)
    resp = post_compare(stereo, cand)
    assert resp.status_code == 200
    data = resp.json()
    assert data["offset_samples"] == 100
    assert data["reference_unmatched_samples"] == 100
    assert data["candidate_unmatched_samples"] == 100
    assert data["overlap_gain_only"] is True


def test_period2_single_sample_edit_picks_zero_without_per_peak_dot():
    # Period-2 alternation (~+/-10000 LSB) on both channels: EVERY lag has
    # |correlation| a hair under 1 (one sample of one channel is edited by
    # 100 LSB in the candidate), so there is no exact +/-1 peak to bail
    # out on and the old implementation ran one full-length dot product per
    # near peak (~9601 of them). The batched exact re-check must keep this
    # fast while still selecting offset 0 by the true exact correlations.
    n = 10 * RATE
    pattern = np.empty(n, dtype="<i2")
    pattern[0::2] = 10000
    pattern[1::2] = -10000
    ref = np.stack([pattern, np.roll(pattern, 1)], axis=1)
    cand = ref.copy()
    cand[n // 2, 0] += 100  # one sample of one channel, middle of file

    import app.compare_gain as cg

    calls = {"n": 0}
    original = cg._exact_overlap_stats

    def counting(ref_int, cand_int, lag):
        calls["n"] += 1
        return original(ref_int, cand_int, lag)

    cg._exact_overlap_stats = counting
    try:
        resp = post_compare(ref, cand)
    finally:
        cg._exact_overlap_stats = original
    assert resp.status_code == 200
    data = resp.json()
    assert data["offset_samples"] == 0
    assert data["reference_unmatched_samples"] == 0
    assert data["candidate_unmatched_samples"] == 0
    assert data["gain_db"] == pytest.approx(0.0, abs=1e-6)
    assert data["polarity_inverted"] is False
    assert data["anomaly_intervals"] == []
    assert data["overlap_gain_only"] is True
    # The whole lag range is resolved by the batched exact FFT; the direct
    # full-overlap dot product fallback must never fire here.
    assert calls["n"] == 0


def test_cross_sum_certificate_fallback_stays_exact(monkeypatch):
    # If a component convolution could not be certified to round exactly,
    # the uncertified lags are recomputed with direct integer dot products;
    # results must stay exact. Force the certificate to reject every lag.
    from app import compare_gain as cg

    n = RATE
    pattern = np.empty(n, dtype="<i2")
    pattern[0::2] = 10000
    pattern[1::2] = -10000
    ref_pcm = np.stack([pattern, np.roll(pattern, 1)], axis=1)
    cand_pcm = ref_pcm.copy()
    cand_pcm[n // 2, 0] += 100
    ref_int = np.rint(ref_pcm.astype(np.float64)).astype(np.int64)
    cand_int = np.rint(cand_pcm.astype(np.float64)).astype(np.int64)
    lags = np.arange(-cg.MAX_SHIFT_SAMPLES, cg.MAX_SHIFT_SAMPLES + 1, dtype=np.int64)

    direct = np.array(
        [cg._exact_overlap_stats(ref_int, cand_int, int(d))[0] for d in lags],
        dtype=np.int64,
    )

    # Baseline: the FFT certifies every lag, results are exact.
    assert np.array_equal(cg._exact_cross_sums(ref_int, cand_int, lags), direct)

    calls = {"n": 0}
    original = cg._exact_overlap_stats

    def counting(r, c, lag):
        calls["n"] += 1
        return original(r, c, lag)

    def reject_all(values, bound):
        return np.zeros(values.shape[0], dtype=bool)

    monkeypatch.setattr(cg, "_exact_overlap_stats", counting)
    monkeypatch.setattr(cg, "_roundoff_certified", reject_all)
    got = cg._exact_cross_sums(ref_int, cand_int, lags)
    assert calls["n"] == lags.shape[0]  # every lag fell back
    assert np.array_equal(got, direct)


def test_fully_periodic_rotation_true_tie_keeps_zero_lag():
    # Pure period with no marker: a one-period rotation correlates at
    # exactly 1 at lag 0 *and* at the rotation lag, so the genuine tie is
    # still broken by the smallest absolute lag (0).
    t = np.arange(RATE) / RATE
    pcm = np.round(10000.0 * np.sin(2 * np.pi * (RATE / 100) * t)).astype("<i2")
    cand = np.roll(pcm, 100)
    resp = post_compare(pcm, cand)
    assert resp.status_code == 200
    data = resp.json()
    assert data["offset_samples"] == 0
    assert data["overlap_gain_only"] is True
