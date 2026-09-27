"""Cut & splice: delete half-open sample intervals and crossfade the seams.

The regions of the input outside ``remove_intervals`` are concatenated in
their original order. At every seam the last F frames of the left segment
and the first F frames of the right segment are overlapped and mixed on
every channel with linear weights: the right segment fades in from 0 to 1
(endpoints included) while the left segment fades out from 1 to 0, where
F = 48 * crossfade_ms frames. F = 0 is a plain hard splice. With seams,
the first/last kept segments must hold at least F frames and every middle
segment at least 2F frames, so the two fades of a middle segment never
share a sample. The output keeps the input format and may be shorter
than one second.

Optional cue intervals on the input sample axis are mapped onto the
spliced output: every cue is intersected with the kept segments and each
surviving fragment is reported with its source interval and its interval
on the output timeline. Kept segment j (0-based) starts at output frame
(kept frames before it) - j*F, so the two fragments flanking a crossfade
zone map to overlapping output intervals; both are kept as-is.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .loudness import (
    SAMPLE_RATE,
    AudioValidationError,
    _quantize,
    _read_wav,
    _write_wav,
)

# 48 kHz -> 48 frames per millisecond of crossfade.
FRAMES_PER_MS = SAMPLE_RATE // 1000


@dataclass
class CueInterval:
    """A named half-open interval on the input sample axis."""

    id: str
    start: int
    end: int


@dataclass
class CueFragment:
    """The part of a cue inside one kept segment, on both timelines."""

    source_interval: tuple[int, int]
    output_interval: tuple[int, int]


@dataclass
class MappedCue:
    """A cue and its surviving fragments, in ascending source order."""

    id: str
    fragments: list[CueFragment] = field(default_factory=list)


@dataclass
class CutSpliceResult:
    wav_bytes: bytes
    output_frames: int
    mapped_cues: list[MappedCue] = field(default_factory=list)


def _validate_intervals(
    remove_intervals: list[tuple[int, int]], n_frames: int
) -> list[tuple[int, int]]:
    """Check the half-open deletion intervals and return them as plain ints.

    Every interval must be non-empty and inside [0, n_frames]; the list
    must be strictly ascending with non-touching neighbours (a gap of at
    least one kept frame between consecutive intervals).
    """
    intervals: list[tuple[int, int]] = []
    for start, end in remove_intervals:
        if not 0 <= start < end <= n_frames:
            raise AudioValidationError(
                f"Interval [{start}, {end}) is empty or outside "
                f"[0, {n_frames})"
            )
        if intervals and start <= intervals[-1][1]:
            raise AudioValidationError(
                "remove_intervals must be strictly ascending and non-touching"
            )
        intervals.append((start, end))
    return intervals


def _validate_cues(cues: list[CueInterval], n_frames: int) -> None:
    """Check cue ids are unique and every interval is non-empty/in bounds."""
    seen: set[str] = set()
    for cue in cues:
        if cue.id in seen:
            raise AudioValidationError(f"Duplicate cue id: {cue.id!r}")
        seen.add(cue.id)
        if not 0 <= cue.start < cue.end <= n_frames:
            raise AudioValidationError(
                f"Cue interval [{cue.start}, {cue.end}) is empty or outside "
                f"[0, {n_frames})"
            )


def _kept_bounds(
    n_frames: int, intervals: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Half-open [start, end) bounds of the kept regions, in order."""
    bounds: list[tuple[int, int]] = []
    cursor = 0
    for start, end in intervals:
        if start > cursor:
            bounds.append((cursor, start))
        cursor = end
    if cursor < n_frames:
        bounds.append((cursor, n_frames))
    return bounds


def _splice(segments: list[np.ndarray], fade_frames: int) -> np.ndarray:
    """Concatenate the kept segments, crossfading F frames at every seam."""
    if fade_frames == 0 or len(segments) == 1:
        return np.concatenate(segments, axis=0)
    # Right-segment weight per overlap frame: 0 -> 1, endpoints included.
    weight = np.linspace(0.0, 1.0, fade_frames).reshape(-1, 1)
    pieces: list[np.ndarray] = []
    prev = segments[0]
    for seg in segments[1:]:
        pieces.append(prev[:-fade_frames])
        pieces.append(
            prev[-fade_frames:] * (1.0 - weight) + seg[:fade_frames] * weight
        )
        prev = seg[fade_frames:]
    pieces.append(prev)
    return np.concatenate(pieces, axis=0)


def _map_cues(
    cues: list[CueInterval],
    kept_bounds: list[tuple[int, int]],
    fade_frames: int,
) -> list[MappedCue]:
    """Map every cue interval onto the spliced output timeline.

    Kept segment j (0-based) starts at output frame (kept frames before
    it) - j*F, so a source frame x inside segment [s, e) lands at
    x + out_start_j - s. The two fragments flanking a crossfade zone
    therefore map to output intervals that overlap by up to F frames;
    both are reported as-is, never merged. Cues are sorted by start and
    swept against the ascending segments in one pass, so the total work
    is O(cues log cues + segments + fragments) instead of cues x
    segments. The result is restored to request order.
    """
    # Output start of every kept segment: cumulative kept frames minus
    # one F per seam already crossed.
    out_starts: list[int] = []
    out_start = 0
    for seg_start, seg_end in kept_bounds:
        out_starts.append(out_start)
        out_start += (seg_end - seg_start) - fade_frames

    fragments: list[list[CueFragment]] = [[] for _ in cues]
    order = sorted(range(len(cues)), key=lambda i: (cues[i].start, i))
    seg_idx = 0
    n_seg = len(kept_bounds)
    for i in order:
        cue = cues[i]
        # Segments ending at or before the cue start intersect neither
        # this cue nor any later one (cue starts are sorted ascending).
        while seg_idx < n_seg and kept_bounds[seg_idx][1] <= cue.start:
            seg_idx += 1
        j = seg_idx
        while j < n_seg and kept_bounds[j][0] < cue.end:
            seg_start, seg_end = kept_bounds[j]
            lo = max(cue.start, seg_start)
            hi = min(cue.end, seg_end)
            if lo < hi:
                shift = out_starts[j] - seg_start
                fragments[i].append(
                    CueFragment(
                        source_interval=(lo, hi),
                        output_interval=(lo + shift, hi + shift),
                    )
                )
            j += 1
    return [
        MappedCue(id=cue.id, fragments=fragments[i])
        for i, cue in enumerate(cues)
    ]


def cut_splice_wav(
    wav_bytes: bytes,
    remove_intervals: list[tuple[int, int]],
    crossfade_ms: int,
    cue_intervals: list[CueInterval] | None = None,
) -> CutSpliceResult:
    samples = _read_wav(wav_bytes)
    n_frames, channels = samples.shape
    fade_frames = FRAMES_PER_MS * crossfade_ms

    intervals = _validate_intervals(remove_intervals, n_frames)
    cues = list(cue_intervals) if cue_intervals else []
    _validate_cues(cues, n_frames)

    bounds = _kept_bounds(n_frames, intervals)
    segments = [samples[start:end] for start, end in bounds]
    if not segments:
        raise AudioValidationError("remove_intervals delete the entire file")

    if fade_frames > 0 and len(segments) > 1:
        # Every seam borrows F tail frames from its left segment and F head
        # frames from its right one; a middle segment feeds two seams, so it
        # needs 2F frames to keep the two fades from sharing samples.
        for index, seg in enumerate(segments):
            edge = index == 0 or index == len(segments) - 1
            required = fade_frames if edge else 2 * fade_frames
            if seg.shape[0] < required:
                raise AudioValidationError(
                    f"Kept segment {index} has {seg.shape[0]} frames but the "
                    f"crossfade needs at least {required}"
                )

    spliced = _splice(segments, fade_frames)
    pcm = _quantize(spliced)
    return CutSpliceResult(
        wav_bytes=_write_wav(pcm, channels),
        output_frames=int(pcm.shape[0]),
        mapped_cues=_map_cues(cues, bounds, fade_frames),
    )
