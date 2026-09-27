"""FastAPI service: audio POST endpoints backed by a shared bounded process
executor. Every route awaits its computation through ``_run_audio``, which
watches the ASGI ``http.disconnect`` event while the worker runs: when a
client closes its TCP connection mid-request, its waiting task is cancelled so
the executor can free the corresponding queue slot (running jobs keep their
worker slot until the computation finishes — only the result is discarded)."""
from __future__ import annotations

import asyncio
import base64
import binascii
import math
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .align_stereo import align_stereo_wav
from .compare_gain import compare_wav_gain
from .compress import compress_wav
from .convolution_reverb import convolution_reverb_wav
from .cut_splice import CueInterval, cut_splice_wav
from .denoise import denoise_wav
from .eq import EqBand, eq_wav
from .executor import BoundedProcessExecutor, ExecutorOverloadedError
from .loudness import SAMPLE_RATE, AudioValidationError, normalize_wav
from .onsets import detect_onsets
from .repair_gaps import RepairMethod, repair_gaps_wav
from .time_stretch import time_stretch_wav


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The shared bounded process executor is created at startup and reclaimed
    # at shutdown; /health never touches it.
    executor = BoundedProcessExecutor.from_env()
    app.state.executor = executor
    try:
        yield
    finally:
        app.state.executor = None
        await asyncio.to_thread(executor.shutdown)


app = FastAPI(title="WAV Loudness Normalizer", version="1.0.0", lifespan=lifespan)


def _get_executor() -> BoundedProcessExecutor:
    executor = getattr(app.state, "executor", None)
    if executor is None or executor.closed:
        # Embedded use without the lifespan (e.g. tests): create lazily.
        executor = BoundedProcessExecutor.from_env()
        app.state.executor = executor
    return executor


class ClientDisconnected(Exception):
    """The client closed its TCP connection while the computation was queued
    or running. Nothing can be delivered anymore; mapped to ``499`` so Uvicorn
    neither logs a 500 nor (with no response started) writes one to a socket
    that is already gone."""


async def _run_audio(http_request: Request, fn, *args):
    """Run one audio computation while watching the client connection.

    The computation runs in a child task so that an ``http.disconnect`` event
    can cancel the executor wait instead of leaving the slot occupied. A
    queued job leaves the queue immediately; a running job keeps its worker
    slot until the computation finishes and only its result is discarded. If
    the computation and the disconnect finish racing, the result wins — the
    executor guarantees each slot is released exactly once either way.
    """

    async def watch_disconnect() -> None:
        # FastAPI already consumed the request body before invoking the route,
        # so the only further ASGI message is http.disconnect (a second
        # http.request cannot arrive here).
        while True:
            message = await http_request.receive()
            if message["type"] == "http.disconnect":
                return

    compute = asyncio.ensure_future(_get_executor().run(fn, *args))
    watcher = asyncio.ensure_future(watch_disconnect())
    try:
        done, _pending = await asyncio.wait(
            {compute, watcher}, return_when=asyncio.FIRST_COMPLETED
        )
    except asyncio.CancelledError:
        # The route task itself is being cancelled (e.g. ASGI shutdown or an
        # in-process cancelled request): tear both children down so the
        # executor wait is not orphaned holding a slot.
        compute.cancel()
        watcher.cancel()
        raise
    if compute in done:
        # Computation completed first (result or business exception); the
        # disconnect watcher is no longer needed.
        watcher.cancel()
        return compute.result()
    if not watcher.cancelled() and watcher.exception() is not None:
        # The ASGI receive failed: cancel the wait and surface the error.
        compute.cancel()
        raise watcher.exception()
    # http.disconnect arrived first: cancel the executor wait. A queued job
    # frees its slot now; a running job is only marked cancelled and releases
    # its worker slot when the computation ends.
    compute.cancel()
    try:
        await compute
    except asyncio.CancelledError:
        pass
    raise ClientDisconnected


def _json_safe(value):
    """Make validation-error details JSON-safe: non-finite floats -> strings."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # Same 422 body as the default handler, but with non-finite floats in
    # the echoed input replaced by strings: json.dumps(allow_nan=False)
    # cannot serialize them and the default handler would turn the
    # validation failure into a 500.
    return JSONResponse(
        status_code=422,
        content={"detail": _json_safe(jsonable_encoder(exc.errors()))},
    )


@app.exception_handler(ExecutorOverloadedError)
async def executor_overloaded_handler(
    request: Request, exc: ExecutorOverloadedError
) -> JSONResponse:
    # Every worker and queue slot is taken: reject immediately instead of
    # piling work into the process pool.
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Audio processing is at capacity; retry after 1 second"
        },
        headers={"Retry-After": "1"},
    )


@app.exception_handler(ClientDisconnected)
async def client_disconnected_handler(
    request: Request, exc: ClientDisconnected
) -> JSONResponse:
    # The client closed the connection during the computation: the slot has
    # already been released (queue) or will be when the worker finishes.
    # 499 = Nginx convention "client closed request"; Uvicorn's send() is a
    # no-op on a closed socket, so this is only observed in-process.
    return JSONResponse(status_code=499, content={"detail": "Client disconnected"})


class NormalizeRequest(BaseModel):
    wav_base64: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded WAV: 48 kHz, 16-bit PCM, mono/stereo, 1-10 s",
    )
    target_lufs: float = Field(
        ..., ge=-30.0, le=-10.0, description="Target integrated loudness (LUFS)"
    )
    ceiling_dbtp: float = Field(
        ..., ge=-6.0, le=-0.5, description="True-peak ceiling (dBTP)"
    )


class NormalizeResponse(BaseModel):
    wav_base64: str = Field(..., description="Normalized WAV, same format, Base64")
    input_loudness_lufs: float
    output_loudness_lufs: float
    input_true_peak_dbtp: float
    output_true_peak_dbtp: float
    gain_db: float = Field(..., description="Static gain actually applied")
    target_reached: bool = Field(
        ..., description="True if output loudness is within 0.1 LU of the target"
    )


class CompareGainRequest(BaseModel):
    reference_wav_base64: str = Field(
        ...,
        min_length=1,
        description="Base64 WAV: 48 kHz, 16-bit PCM, mono/stereo, 1-10 s",
    )
    candidate_wav_base64: str = Field(
        ...,
        min_length=1,
        description="Base64 WAV: same format and channel count as the reference",
    )


class CompareGainResponse(BaseModel):
    offset_samples: int = Field(
        ...,
        description="Candidate index minus reference index for the aligned "
        "samples; positive means the candidate is delayed vs. the reference",
    )
    gain_db: float = Field(
        ..., description="Magnitude of the single least-squares gain (dB)"
    )
    polarity_inverted: bool = Field(
        ..., description="True if the least-squares gain was negative"
    )
    reference_unmatched_samples: int = Field(
        ..., description="Reference frames outside the aligned overlap"
    )
    candidate_unmatched_samples: int = Field(
        ..., description="Candidate frames outside the aligned overlap"
    )
    anomaly_intervals: list[tuple[float, float]] = Field(
        ...,
        description="Half-open [start_ms, end_ms) intervals on the reference "
        "timeline whose residual exceeds the per-window threshold",
    )
    overlap_gain_only: bool = Field(
        ..., description="True when no window has an above-threshold residual"
    )


class CueIntervalModel(BaseModel):
    id: str = Field(..., min_length=1, description="Unique non-empty cue id")
    start: int = Field(..., description="Cue start frame (inclusive)")
    end: int = Field(..., description="Cue end frame (exclusive)")


class CutSpliceRequest(BaseModel):
    wav_base64: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded WAV: 48 kHz, 16-bit PCM, mono/stereo, 1-10 s",
    )
    remove_intervals: list[tuple[int, int]] = Field(
        ...,
        description="Half-open [start, end) deletion intervals as input "
        "sample indices; strictly ascending, non-touching, in bounds",
    )
    crossfade_ms: int = Field(
        ...,
        ge=0,
        le=20,
        description="Seam crossfade length in milliseconds; 0 = hard splice",
    )
    cue_intervals: list[CueIntervalModel] | None = Field(
        default=None,
        max_length=2000,
        description="Optional cue intervals on the input sample axis; "
        "half-open [start, end), in bounds, unique non-empty ids; "
        "any order, overlaps allowed",
    )


class CueFragmentModel(BaseModel):
    source_interval: tuple[int, int] = Field(
        ..., description="Half-open fragment on the input sample axis"
    )
    output_interval: tuple[int, int] = Field(
        ..., description="The same fragment on the spliced output timeline"
    )


class MappedCueModel(BaseModel):
    id: str = Field(..., description="Cue id from the request")
    fragments: list[CueFragmentModel] = Field(
        ..., description="Surviving fragments, ascending source order; "
        "empty when the cue is fully deleted",
    )


class CutSpliceResponse(BaseModel):
    wav_base64: str = Field(..., description="Spliced WAV, same format, Base64")
    output_frames: int = Field(
        ...,
        description="Output frames: kept frames minus seams x fade frames",
    )
    mapped_cues: list[MappedCueModel] = Field(
        ...,
        description="Requested cues mapped onto the output timeline, in "
        "request order; empty when cue_intervals was not provided",
    )


class RepairGapsRequest(BaseModel):
    wav_base64: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded WAV: 48 kHz, 16-bit PCM, mono/stereo, 1-10 s",
    )
    gaps: list[tuple[int, int]] = Field(
        ...,
        description="Half-open [start, end) zero-sample gaps as input sample "
        "indices; strictly ascending, non-overlapping; each 1-20 ms long, at "
        "least 40 ms from either edge and 60 ms apart",
    )


class GapRepairModel(BaseModel):
    start: int = Field(..., description="Gap start sample index (inclusive)")
    end: int = Field(..., description="Gap end sample index (exclusive)")
    method: RepairMethod = Field(
        ..., description="predicted: 24th-order LPC cross-faded; "
        "linear: boundary-linear fallback"
    )


class RepairGapsResponse(BaseModel):
    wav_base64: str = Field(..., description="Repaired WAV, same format, Base64")
    repairs: list[GapRepairModel] = Field(
        ..., description="Repair method actually used per gap, in input order"
    )


class DenoiseRequest(BaseModel):
    wav_base64: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded WAV: 48 kHz, 16-bit PCM, mono/stereo, 1-10 s",
    )
    noise_interval: tuple[int, int] = Field(
        ...,
        description="Half-open [start, end) noise-only reference as input "
        "sample indices; 0.2-2 s long, non-zero on every channel",
    )
    strength: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Suppression strength: 0 leaves the PCM untouched, "
        "1 applies full spectral subtraction",
    )


class DenoiseResponse(BaseModel):
    wav_base64: str = Field(
        ...,
        description="Denoised WAV, same frame count, channels and format, Base64",
    )


class CompressRequest(BaseModel):
    wav_base64: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded WAV: 48 kHz, 16-bit PCM, mono/stereo, 1-10 s",
    )
    threshold_dbfs: float = Field(
        ..., ge=-40.0, le=-6.0, description="Compression threshold (dBFS)"
    )
    ratio: float = Field(
        ..., ge=1.0, le=20.0, description="Compression ratio: 1 = no compression"
    )
    attack_ms: float = Field(
        ..., ge=1.0, le=50.0, description="Envelope attack time (ms)"
    )
    release_ms: float = Field(
        ..., ge=10.0, le=500.0, description="Envelope release time (ms)"
    )
    knee_db: float = Field(
        ..., ge=0.0, le=12.0, description="Soft-knee width (dB); 0 = hard knee"
    )


class CompressResponse(BaseModel):
    wav_base64: str = Field(
        ...,
        description="Compressed WAV, same frame count, channels and format, Base64",
    )


class TimeStretchRequest(BaseModel):
    wav_base64: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded WAV: 48 kHz, 16-bit PCM, mono/stereo, 1-10 s",
    )
    speed: float = Field(
        ...,
        ge=0.8,
        le=1.25,
        description="Time-scale factor: 0.8-1.25; duration changes by 1/speed, "
        "pitch is preserved; 1 leaves the PCM unchanged",
    )


class TimeStretchResponse(BaseModel):
    wav_base64: str = Field(
        ...,
        description="Time-stretched WAV, same sample rate, bit depth and "
        "channels, Base64",
    )
    output_frames: int = Field(
        ..., description="Output frame count: round(input_frames / speed)"
    )


class AlignStereoRequest(BaseModel):
    wav_base64: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded WAV: 48 kHz, 16-bit PCM, stereo, 1-10 s",
    )
    calibration_interval: tuple[int, int] = Field(
        ...,
        description="Half-open [start, end) calibration interval as input "
        "sample indices; the two tracks are assumed to carry the same "
        "source there; in bounds, 0.2-2 s long",
    )


class AlignStereoResponse(BaseModel):
    wav_base64: str = Field(
        ...,
        description="Aligned WAV, same frame count, channels and format, "
        "Base64",
    )
    offset_samples: int = Field(
        ...,
        description="Right-track index minus left-track index for "
        "corresponding samples; positive means the right track is delayed",
    )
    polarity_inverted: bool = Field(
        ..., description="True if the winning correlation was negative and "
        "the right track was inverted"
    )
    correlation: float = Field(
        ..., description="Signed normalized cross-correlation of the "
        "calibration segment at the chosen offset"
    )


class EqBandModel(BaseModel):
    frequency_hz: float = Field(
        ...,
        ge=40.0,
        le=16000.0,
        allow_inf_nan=False,
        description="Band centre frequency (Hz)",
    )
    q: float = Field(
        ..., ge=0.3, le=10.0, allow_inf_nan=False, description="Band Q factor"
    )
    gain_db: float = Field(
        ...,
        ge=-12.0,
        le=12.0,
        allow_inf_nan=False,
        description="Peak gain at the centre frequency (dB)",
    )


class EqRequest(BaseModel):
    wav_base64: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded WAV: 48 kHz, 16-bit PCM, mono/stereo, 1-10 s",
    )
    bands: list[EqBandModel] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="Peaking-EQ bands, cascaded in the given order",
    )
    ceiling_dbtp: float = Field(
        ...,
        ge=-6.0,
        le=-0.5,
        allow_inf_nan=False,
        description="True-peak ceiling (dBTP)",
    )


class OnsetsRequest(BaseModel):
    wav_base64: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded WAV: 48 kHz, 16-bit PCM, mono/stereo, 1-10 s",
    )
    min_relative_flux: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum spectral flux of a reported onset, as a "
        "fraction of the whole-file maximum flux",
    )


class OnsetModel(BaseModel):
    frame_index: int = Field(
        ..., description="Window-centre sample index, clamped to [0, n-1]"
    )
    strength: float = Field(
        ..., description="Spectral flux relative to the whole-file maximum"
    )


class OnsetsResponse(BaseModel):
    onsets: list[OnsetModel] = Field(
        ..., description="Detected onsets in ascending time order"
    )


class EqResponse(BaseModel):
    wav_base64: str = Field(
        ..., description="EQ'd WAV, same frame count and format, Base64"
    )
    input_true_peak_dbtp: float
    output_true_peak_dbtp: float
    post_gain_db: float = Field(
        ..., description="Uniform post-filter gain actually applied (dB); "
        "never positive"
    )


class ConvolutionReverbRequest(BaseModel):
    wav_base64: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded WAV: 48 kHz, 16-bit PCM, mono/stereo, 1-10 s",
    )
    ir_wav_base64: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded impulse-response WAV: 48 kHz, 16-bit "
        "PCM, mono/stereo, 1-2 s, not all silence",
    )
    wet: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Wet mix: 0 keeps the zero-padded dry signal, "
        "1 is the pure convolution",
    )


class ConvolutionReverbResponse(BaseModel):
    wav_base64: str = Field(
        ...,
        description="Reverbed WAV: 48 kHz, 16-bit PCM, N+M-1 frames, Base64",
    )
    output_frames: int = Field(
        ..., description="Output frame count: source frames + IR frames - 1"
    )
    post_gain_db: float = Field(
        ..., description="Uniform gain actually applied to all channels "
        "(dB); never positive, 0 when no attenuation was needed"
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/normalize", response_model=NormalizeResponse)
async def normalize(
    http_request: Request, request: NormalizeRequest
) -> NormalizeResponse:
    try:
        wav_bytes = base64.b64decode(request.wav_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"Invalid Base64 payload: {exc}"
        ) from exc

    try:
        result = await _run_audio(
            http_request,
            normalize_wav,
            wav_bytes,
            request.target_lufs,
            request.ceiling_dbtp,
        )
    except AudioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return NormalizeResponse(
        wav_base64=base64.b64encode(result.wav_bytes).decode("ascii"),
        input_loudness_lufs=round(result.input_loudness_lufs, 3),
        output_loudness_lufs=round(result.output_loudness_lufs, 3),
        input_true_peak_dbtp=round(result.input_true_peak_dbtp, 3),
        output_true_peak_dbtp=round(result.output_true_peak_dbtp, 3),
        gain_db=round(result.gain_db, 3),
        target_reached=result.target_reached,
    )


@app.post("/v1/compare-gain", response_model=CompareGainResponse)
async def compare_gain(
    http_request: Request, request: CompareGainRequest
) -> CompareGainResponse:
    try:
        reference_wav = base64.b64decode(
            request.reference_wav_base64, validate=True
        )
        candidate_wav = base64.b64decode(
            request.candidate_wav_base64, validate=True
        )
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"Invalid Base64 payload: {exc}"
        ) from exc

    try:
        result = await _run_audio(
            http_request, compare_wav_gain, reference_wav, candidate_wav
        )
    except AudioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    ms_per_sample = 1000.0 / SAMPLE_RATE
    return CompareGainResponse(
        offset_samples=result.offset_samples,
        gain_db=round(result.gain_db, 3),
        polarity_inverted=result.polarity_inverted,
        reference_unmatched_samples=result.reference_unmatched_samples,
        candidate_unmatched_samples=result.candidate_unmatched_samples,
        anomaly_intervals=[
            (round(start * ms_per_sample, 3), round(end * ms_per_sample, 3))
            for start, end in result.anomaly_intervals
        ],
        overlap_gain_only=result.overlap_gain_only,
    )


@app.post("/v1/cut-splice", response_model=CutSpliceResponse)
async def cut_splice(
    http_request: Request, request: CutSpliceRequest
) -> CutSpliceResponse:
    try:
        wav_bytes = base64.b64decode(request.wav_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"Invalid Base64 payload: {exc}"
        ) from exc

    try:
        result = await _run_audio(
            http_request,
            cut_splice_wav,
            wav_bytes,
            request.remove_intervals,
            request.crossfade_ms,
            [
                CueInterval(id=cue.id, start=cue.start, end=cue.end)
                for cue in request.cue_intervals or []
            ],
        )
    except AudioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return CutSpliceResponse(
        wav_base64=base64.b64encode(result.wav_bytes).decode("ascii"),
        output_frames=result.output_frames,
        mapped_cues=[
            MappedCueModel(
                id=cue.id,
                fragments=[
                    CueFragmentModel(
                        source_interval=fragment.source_interval,
                        output_interval=fragment.output_interval,
                    )
                    for fragment in cue.fragments
                ],
            )
            for cue in result.mapped_cues
        ],
    )


@app.post("/v1/repair-gaps", response_model=RepairGapsResponse)
async def repair_gaps(
    http_request: Request, request: RepairGapsRequest
) -> RepairGapsResponse:
    try:
        wav_bytes = base64.b64decode(request.wav_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"Invalid Base64 payload: {exc}"
        ) from exc

    try:
        result = await _run_audio(
            http_request, repair_gaps_wav, wav_bytes, request.gaps
        )
    except AudioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return RepairGapsResponse(
        wav_base64=base64.b64encode(result.wav_bytes).decode("ascii"),
        repairs=[
            GapRepairModel(start=r.start, end=r.end, method=r.method)
            for r in result.repairs
        ],
    )


@app.post("/v1/denoise", response_model=DenoiseResponse)
async def denoise(
    http_request: Request, request: DenoiseRequest
) -> DenoiseResponse:
    try:
        wav_bytes = base64.b64decode(request.wav_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"Invalid Base64 payload: {exc}"
        ) from exc

    try:
        result = await _run_audio(
            http_request,
            denoise_wav,
            wav_bytes,
            request.noise_interval,
            request.strength,
        )
    except AudioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return DenoiseResponse(
        wav_base64=base64.b64encode(result.wav_bytes).decode("ascii"),
    )


@app.post("/v1/compress", response_model=CompressResponse)
async def compress(
    http_request: Request, request: CompressRequest
) -> CompressResponse:
    try:
        wav_bytes = base64.b64decode(request.wav_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"Invalid Base64 payload: {exc}"
        ) from exc

    try:
        result = await _run_audio(
            http_request,
            compress_wav,
            wav_bytes,
            request.threshold_dbfs,
            request.ratio,
            request.attack_ms,
            request.release_ms,
            request.knee_db,
        )
    except AudioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return CompressResponse(
        wav_base64=base64.b64encode(result.wav_bytes).decode("ascii"),
    )


@app.post("/v1/time-stretch", response_model=TimeStretchResponse)
async def time_stretch(
    http_request: Request, request: TimeStretchRequest
) -> TimeStretchResponse:
    try:
        wav_bytes = base64.b64decode(request.wav_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"Invalid Base64 payload: {exc}"
        ) from exc

    try:
        result = await _run_audio(
            http_request, time_stretch_wav, wav_bytes, request.speed
        )
    except AudioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return TimeStretchResponse(
        wav_base64=base64.b64encode(result.wav_bytes).decode("ascii"),
        output_frames=result.output_frames,
    )


@app.post("/v1/align-stereo", response_model=AlignStereoResponse)
async def align_stereo(
    http_request: Request, request: AlignStereoRequest
) -> AlignStereoResponse:
    try:
        wav_bytes = base64.b64decode(request.wav_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"Invalid Base64 payload: {exc}"
        ) from exc

    try:
        result = await _run_audio(
            http_request, align_stereo_wav, wav_bytes, request.calibration_interval
        )
    except AudioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return AlignStereoResponse(
        wav_base64=base64.b64encode(result.wav_bytes).decode("ascii"),
        offset_samples=result.offset_samples,
        polarity_inverted=result.polarity_inverted,
        correlation=round(result.correlation, 6),
    )


@app.post("/v1/eq", response_model=EqResponse)
async def eq(http_request: Request, request: EqRequest) -> EqResponse:
    try:
        wav_bytes = base64.b64decode(request.wav_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"Invalid Base64 payload: {exc}"
        ) from exc

    try:
        result = await _run_audio(
            http_request,
            eq_wav,
            wav_bytes,
            [
                EqBand(
                    frequency_hz=band.frequency_hz,
                    q=band.q,
                    gain_db=band.gain_db,
                )
                for band in request.bands
            ],
            request.ceiling_dbtp,
        )
    except AudioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return EqResponse(
        wav_base64=base64.b64encode(result.wav_bytes).decode("ascii"),
        input_true_peak_dbtp=round(result.input_true_peak_dbtp, 3),
        output_true_peak_dbtp=round(result.output_true_peak_dbtp, 3),
        post_gain_db=round(result.post_gain_db, 3),
    )


@app.post("/v1/onsets", response_model=OnsetsResponse)
async def onsets(
    http_request: Request, request: OnsetsRequest
) -> OnsetsResponse:
    try:
        wav_bytes = base64.b64decode(request.wav_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"Invalid Base64 payload: {exc}"
        ) from exc

    try:
        result = await _run_audio(
            http_request,
            detect_onsets,
            wav_bytes,
            request.min_relative_flux,
        )
    except AudioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return OnsetsResponse(
        onsets=[
            OnsetModel(frame_index=onset.frame_index, strength=onset.strength)
            for onset in result.onsets
        ]
    )


@app.post("/v1/convolution-reverb", response_model=ConvolutionReverbResponse)
async def convolution_reverb(
    http_request: Request,
    request: ConvolutionReverbRequest,
) -> ConvolutionReverbResponse:
    try:
        wav_bytes = base64.b64decode(request.wav_base64, validate=True)
        ir_wav_bytes = base64.b64decode(request.ir_wav_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"Invalid Base64 payload: {exc}"
        ) from exc

    try:
        result = await _run_audio(
            http_request,
            convolution_reverb_wav,
            wav_bytes,
            ir_wav_bytes,
            request.wet,
        )
    except AudioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ConvolutionReverbResponse(
        wav_base64=base64.b64encode(result.wav_bytes).decode("ascii"),
        output_frames=result.output_frames,
        post_gain_db=round(result.post_gain_db, 3),
    )
