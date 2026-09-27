"""WSOLA time-scale modification: change duration by 1/speed while keeping
the original pitch.

Output frame count is round(input_frames / speed). Analysis segments are
40 ms (SEGMENT_FRAMES = 1920) long and regular output hops are 20 ms
(HOP_FRAMES = 960), so neighbouring segments overlap over 960 frames and
are mixed with a linear cross-fade. The first segment is the very first
40 ms of the input and the last segment is the very last 40 ms. For every
segment in between the nominal input start follows the output position
linearly mapped between the two fixed segments, and the actual start is
picked within +/- SEARCH_FRAMES (240 frames) of it by the normalized
cross-correlation of the overlap against the previous segment, summed
over all channels so a stereo stream moves as one. Ties go to the start
nearest the nominal position and then the smaller index; zero energy in
the reference overlap collapses the search to the nearest position. The
last output hop is shortened as needed to land on the exact output frame
count. The last piece then overlaps the previous one over
SEGMENT_FRAMES - final_hop frames (the previous segment still runs that
long past the last piece's position), so that whole zone - already
holding the mixed output of the previous pieces - is faded against the
last piece's head with a linear 0 -> 1 ramp, and only the last piece's
remaining PCM is copied verbatim. The output keeps the input sample
rate, bit depth and channel count; speed = 1 leaves the PCM
sample-for-sample unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .loudness import (
    SAMPLE_RATE,
    AudioValidationError,
    _quantize,
    _read_wav,
    _write_wav,
)

# 48 kHz -> 48 frames per millisecond.
FRAMES_PER_MS = SAMPLE_RATE // 1000
SEGMENT_FRAMES = 40 * FRAMES_PER_MS  # 40 ms analysis segments
HOP_FRAMES = 20 * FRAMES_PER_MS      # 20 ms regular output hop
SEARCH_FRAMES = 240                  # +/- search around the nominal start
OVERLAP_FRAMES = SEGMENT_FRAMES - HOP_FRAMES  # 960, linear cross-fade zone
MIN_SPEED = 0.8
MAX_SPEED = 1.25


@dataclass
class TimeStretchResult:
    wav_bytes: bytes
    output_frames: int


def _choose_start(
    samples: np.ndarray,
    nominal: int,
    previous_start: int,
    hop: int = HOP_FRAMES,
) -> int:
    """Pick the analysis start of one middle segment.

    The reference overlap is the head of the *previous* chosen piece that
    the new segment fades against: ``hop`` frames beginning at
    ``previous_start + SEGMENT_FRAMES - hop``. Candidate starts are the
    integer positions within +/- SEARCH_FRAMES of ``nominal`` that keep the
    whole segment and its ``hop``-frame overlap head inside the input. The
    score is the normalized cross-correlation of the overlap, summed over
    channels before normalization. A zero-energy reference overlap (or no
    energy in any candidate overlap) skips correlation and takes the
    position nearest ``nominal``; exact score ties break the same way, then
    by the smaller index.
    """
    n_frames = samples.shape[0]
    ref_start = previous_start + SEGMENT_FRAMES - hop
    reference = samples[ref_start : ref_start + hop]  # (hop, channels)

    # Candidate starts: +/- SEARCH_FRAMES around the nominal position,
    # clipped to starts whose whole segment fits in the input.
    low = max(0, nominal - SEARCH_FRAMES)
    high = min(nominal + SEARCH_FRAMES, n_frames - SEGMENT_FRAMES)
    if low > high:  # pragma: no cover - unreachable for valid 1-10 s input
        raise AudioValidationError("Time-stretch search window is empty")

    starts = np.arange(low, high + 1)
    nearest = int(starts[np.argmin(np.abs(starts - nominal))])
    if not np.any(reference):
        # Nothing to correlate against: use the nearest position directly.
        return nearest

    # sliding_window_view appends a window axis at the end; fancy indexing
    # by candidate starts gives (candidates, channels, hop), moved to
    # (candidates, hop, channels) to match the reference layout.
    windows = np.lib.stride_tricks.sliding_window_view(samples, hop, axis=0)[
        starts
    ]
    windows = np.moveaxis(windows, 1, 2)  # (candidates, hop, channels)

    ref_energy = float(np.sum(reference * reference))
    cand_energy = np.einsum("cnk,cnk->c", windows, windows)
    valid = cand_energy > 0.0
    if not np.any(valid):
        return nearest

    dots = np.einsum("cnk,nk->c", windows[valid], reference)
    scores = np.full(starts.shape[0], -np.inf, dtype=np.float64)
    scores[valid] = dots / np.sqrt(cand_energy[valid] * ref_energy)

    distance = np.abs(starts - nominal)
    order = sorted(
        range(starts.shape[0]), key=lambda i: (-scores[i], distance[i], i)
    )
    return int(starts[order[0]])


def _place_segments(
    samples: np.ndarray, output_frames: int
) -> tuple[list[int], list[tuple[int, int]]]:
    """Choose every segment's input start and its output (position, hop).

    The first segment is the input head, the last the input tail; middle
    segments sit at regular HOP_FRAMES output positions with regular hops,
    and the final hop shortens so the tail segment ends exactly at
    ``output_frames``. Nominal middle starts map the output position
    linearly between the two fixed input segments.
    """
    n_frames = samples.shape[0]
    first_start = 0
    last_start = n_frames - SEGMENT_FRAMES
    last_pos = output_frames - SEGMENT_FRAMES

    # Regular output positions strictly between the two fixed segments.
    middle_positions = list(range(HOP_FRAMES, last_pos, HOP_FRAMES))

    starts: list[int] = [first_start]
    placements: list[tuple[int, int]] = [(0, HOP_FRAMES)]
    previous_start = first_start
    for pos in middle_positions:
        # Nominal input start: output position mapped linearly between the
        # head segment (0 -> 0) and the tail segment (last_pos -> last_start).
        nominal = int(round(last_start * pos / last_pos))
        start = _choose_start(samples, nominal, previous_start)
        starts.append(start)
        placements.append((pos, HOP_FRAMES))
        previous_start = start

    starts.append(last_start)
    # Shorten the final hop so the tail ends exactly at output_frames.
    final_hop = last_pos - placements[-1][0]
    placements.append((last_pos, final_hop))
    return starts, placements


def _overlap_add(
    samples: np.ndarray,
    starts: list[int],
    placements: list[tuple[int, int]],
) -> np.ndarray:
    """Render the segments at their output positions with linear fades.

    Pieces follow one another with output hop ``hop``: each seam overlaps
    the previous piece's last ``hop`` frames with the new piece's first
    ``hop`` frames, the former fading out from 1 to 0 and the latter fading
    in from 0 to 1. Every middle piece is written head to segment end; the
    next seam overwrites the shared part with its own blend, so only the
    part past the next seam survives as the piece's exclusive middle.

    The last piece carries the shortened final hop, so the previous piece's
    segment still extends SEGMENT_FRAMES - hop frames past its position and
    the true overlap is that long, not just ``hop``. The overlap zone
    already holds the mixed output of the previous pieces, and the last
    piece is aligned head to its position; the two are blended with a fresh
    linear 0 -> 1 ramp over the whole zone, after which the last piece's
    remaining PCM is copied and reaches the exact output frame count.
    """
    output_frames = placements[-1][0] + SEGMENT_FRAMES
    output = np.zeros((output_frames, samples.shape[1]), dtype=np.float64)
    fade_in = np.linspace(0.0, 1.0, HOP_FRAMES).reshape(-1, 1)
    fade_out = 1.0 - fade_in

    first_pos = placements[0][0]
    output[first_pos : first_pos + HOP_FRAMES] = samples[0:HOP_FRAMES]
    last_index = len(placements) - 1
    for index, (start, (pos, hop)) in enumerate(
        zip(starts[1:], placements[1:]), start=1
    ):
        if index == last_index:
            # Shortened final hop: fade over the full overlap with the
            # previous piece (SEGMENT_FRAMES - hop), against the output
            # already mixed there, then copy the last piece's tail PCM.
            overlap = SEGMENT_FRAMES - hop
            weights = np.linspace(0.0, 1.0, overlap).reshape(-1, 1)
            output[pos : pos + overlap] = (
                output[pos : pos + overlap] * (1.0 - weights)
                + samples[start : start + overlap] * weights
            )
            output[pos + overlap : pos + SEGMENT_FRAMES] = samples[
                start + overlap : start + SEGMENT_FRAMES
            ]
            continue

        prev_start = starts[index - 1]
        # Seam: previous tail fading out, new head fading in.
        output[pos : pos + hop] = (
            samples[
                prev_start
                + SEGMENT_FRAMES
                - hop : prev_start
                + SEGMENT_FRAMES
            ]
            * fade_out[:hop]
            + samples[start : start + hop] * fade_in[:hop]
        )
        # Rest of the piece up to its segment end; the next seam rewrites
        # the shared part with its blend, so only the part beyond the next
        # seam survives as this piece's exclusive middle. When the next
        # piece is the last one with a shortened hop, its longer overlap
        # reads this tail as the "already mixed" output it fades against.
        output[pos + hop : pos + SEGMENT_FRAMES] = samples[
            start + hop : start + SEGMENT_FRAMES
        ]
    return output


def time_stretch_wav(wav_bytes: bytes, speed: float) -> TimeStretchResult:
    samples = _read_wav(wav_bytes)
    n_frames, channels = samples.shape

    if not MIN_SPEED <= float(speed) <= MAX_SPEED:
        raise AudioValidationError(
            f"speed must be between {MIN_SPEED} and {MAX_SPEED} "
            f"(got {speed})"
        )

    output_frames = int(round(n_frames / float(speed)))

    # speed = 1 must leave the PCM untouched, bypassing float fade math.
    if float(speed) == 1.0:
        pcm = _quantize(samples)
        return TimeStretchResult(
            wav_bytes=_write_wav(pcm, channels), output_frames=n_frames
        )

    starts, placements = _place_segments(samples, output_frames)
    stretched = _overlap_add(samples, starts, placements)
    pcm = _quantize(stretched)
    return TimeStretchResult(
        wav_bytes=_write_wav(pcm, channels), output_frames=output_frames
    )
