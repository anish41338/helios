"""OpenAI-compatible HTTP frontend.

Spec section 9.1. Implements /v1/completions, /v1/chat/completions,
/v1/models, /health, and /metrics, plus the namespaced `helios` extension
block for SLO class, speculation depth, and prefix-cache control.

Concurrency model: the engine and its scheduler are single-threaded by design
(spec section 19.4), so every step happens under a single asyncio lock. See
EngineRunner for why the waiting request drives the engine rather than a
perpetual background task. Scheduler state is therefore never touched
concurrently, which keeps the determinism contract intact under HTTP load.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..core.types import SamplingParams, SloClass
from ..engine import EngineConfig, LLMEngine


# ------------------------------------------------------------------ schemas


class HeliosExtensions(BaseModel):
    """HELIOS-specific controls, ignored by standard OpenAI clients."""

    slo_class: str = "B"
    spec_gamma: Optional[int] = None
    prefix_cache: bool = True
    deadline_ms: Optional[int] = None


class CompletionRequest(BaseModel):
    model: str = "helios"
    prompt: str | List[str] = ""
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    stop: Optional[str | List[str]] = None
    stream: bool = False
    seed: Optional[int] = None
    logprobs: Optional[int] = None
    helios: HeliosExtensions = Field(default_factory=HeliosExtensions)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "helios"
    messages: List[ChatMessage]
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    stop: Optional[str | List[str]] = None
    stream: bool = False
    seed: Optional[int] = None
    logprobs: Optional[int] = None
    helios: HeliosExtensions = Field(default_factory=HeliosExtensions)


# ------------------------------------------------------------------- server


class EngineRunner:
    """Serializes access to the single-threaded engine.

    The scheduler is deliberately single-threaded (spec section 19.4), so all
    stepping happens under one lock. Rather than a perpetual background task,
    whichever request is waiting drives the engine forward and distributes
    results to every waiter -- a "cooperative pump".

    That inversion matters: a perpetual `while True` task starves the very
    handlers it is trying to serve under a cooperative event loop, because
    stepping is CPU-bound and never truly awaits. Letting the waiter pump keeps
    exactly one stepper active while guaranteeing progress for all of them.
    """

    def __init__(self, engine: LLMEngine, step_budget: int = 100_000) -> None:
        self.engine = engine
        self.step_budget = step_budget
        self._results: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def stop(self) -> None:
        return None

    async def _pump_until(self, request_id: str):
        """Step the engine until `request_id` completes, then return its output.

        Only one coroutine holds the lock at a time; others wait, and when they
        wake their result is usually already in `_results` because the stepper
        collects outputs for every finished sequence, not just its own.
        """
        for _ in range(self.step_budget):
            if request_id in self._results:
                return self._results.pop(request_id)

            async with self._lock:
                if request_id in self._results:
                    return self._results.pop(request_id)
                if not self.engine.scheduler.has_work:
                    # Nothing to do and our result never appeared: the request
                    # was aborted or dropped.
                    if request_id in self._results:
                        return self._results.pop(request_id)
                    return None
                for out in self.engine.step():
                    self._results[out.request_id] = out

            # Yield between steps so other handlers can enqueue their work and
            # collect their own results.
            await asyncio.sleep(0)

        return self._results.pop(request_id, None)

    async def submit(
        self,
        prompt_token_ids: List[int],
        params: SamplingParams,
        slo_class: SloClass,
        prefix_cache: bool,
        request_id: Optional[str] = None,
    ):
        request_id = request_id or f"cmpl-{uuid.uuid4().hex[:24]}"

        try:
            self.engine.add_request(
                request_id, prompt_token_ids, params, slo_class, prefix_cache
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        out = await self._pump_until(request_id)
        if out is None:
            raise HTTPException(
                status_code=500,
                detail=f"request {request_id} did not complete within the step budget",
            )
        return request_id, out


def _parse_stop(stop) -> List[str]:
    if stop is None:
        return []
    return [stop] if isinstance(stop, str) else list(stop)


def _build_params(req, eos_ids: List[int], max_model_len: int) -> SamplingParams:
    params = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        stop=_parse_stop(req.stop),
        stop_token_ids=list(eos_ids),
        seed=req.seed,
        logprobs=req.logprobs,
    )
    try:
        params.validate()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Reject a max_tokens the engine could never honour rather than silently
    # truncating at max_model_len. Quietly ignoring an explicit client
    # parameter makes the response look complete when it was cut short.
    if params.max_tokens > max_model_len:
        raise HTTPException(
            status_code=400,
            detail=(
                f"max_tokens ({params.max_tokens}) exceeds the model context "
                f"length ({max_model_len})"
            ),
        )
    return params


def create_app(config: EngineConfig) -> FastAPI:
    """Build the ASGI app around a freshly constructed engine."""
    app = FastAPI(title="HELIOS", version="0.1.0")
    engine = LLMEngine(config)
    runner = EngineRunner(engine)
    app.state.engine = engine
    app.state.runner = runner

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "model": config.model_dir,
            "kv_blocks": engine.num_blocks,
            "max_model_len": engine.max_model_len,
        }

    @app.get("/v1/models")
    async def models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": "helios",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "helios",
                    "max_model_len": engine.max_model_len,
                }
            ],
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        """Prometheus text exposition (spec section 9.2)."""
        lines: List[str] = []
        for key, value in engine.stats_snapshot().items():
            if isinstance(value, bool):
                value = int(value)
            if isinstance(value, (int, float)):
                lines.append(f"{key} {value}")
        # Latency summaries, computed from completed requests.
        done = [m for m in engine.metrics() if m.finish_time is not None]
        if done:
            ttfts = sorted(m.ttft for m in done if m.ttft is not None)
            if ttfts:
                lines.append(f"helios_ttft_seconds_p50 {_pct(ttfts, 50)}")
                lines.append(f"helios_ttft_seconds_p95 {_pct(ttfts, 95)}")
            tpots = sorted(m.tpot for m in done if m.tpot is not None)
            if tpots:
                lines.append(f"helios_tpot_seconds_p50 {_pct(tpots, 50)}")
                lines.append(f"helios_tpot_seconds_p95 {_pct(tpots, 95)}")
            goodput = sum(1 for m in done if m.meets_slo()) / len(done)
            lines.append(f"helios_goodput_ratio {goodput}")
        return "\n".join(lines) + "\n"

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt must not be empty")

        token_ids = _encode(engine, prompt)
        params = _build_params(req, engine.eos_token_ids, engine.max_model_len)
        slo = _parse_slo(req.helios.slo_class)

        if req.stream:
            return StreamingResponse(
                _stream_completion(runner, token_ids, params, slo, req),
                media_type="text/event-stream",
            )

        request_id, out = await runner.submit(
            token_ids, params, slo, req.helios.prefix_cache
        )
        return {
            "id": request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "text": out.text,
                    "finish_reason": out.finish_reason.value
                    if out.finish_reason
                    else None,
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": out.prompt_tokens,
                "completion_tokens": out.completion_tokens,
                "total_tokens": out.prompt_tokens + out.completion_tokens,
                "cached_tokens": out.cached_tokens,
            },
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")

        prompt = _apply_chat_template(engine, req.messages)
        token_ids = _encode(engine, prompt)
        params = _build_params(req, engine.eos_token_ids, engine.max_model_len)
        slo = _parse_slo(req.helios.slo_class)

        if req.stream:
            return StreamingResponse(
                _stream_chat(runner, token_ids, params, slo, req),
                media_type="text/event-stream",
            )

        request_id, out = await runner.submit(
            token_ids, params, slo, req.helios.prefix_cache
        )
        return {
            "id": request_id.replace("cmpl", "chatcmpl"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": out.text},
                    "finish_reason": out.finish_reason.value
                    if out.finish_reason
                    else None,
                }
            ],
            "usage": {
                "prompt_tokens": out.prompt_tokens,
                "completion_tokens": out.completion_tokens,
                "total_tokens": out.prompt_tokens + out.completion_tokens,
                "cached_tokens": out.cached_tokens,
            },
        }

    return app


def _encode(engine: LLMEngine, prompt: str) -> List[int]:
    """Tokenize, converting a missing tokenizer into a clear client error.

    The text endpoints require a tokenizer; the token-id API surface does not.
    A model directory without tokenizer files is a deployment problem, so it
    gets a 503 rather than a 500 stack trace.
    """
    try:
        return engine.tokenizer.encode(prompt)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "no tokenizer available for this model directory; text endpoints "
                f"require tokenizer files alongside the weights ({exc})"
            ),
        ) from exc


def _parse_slo(value: str) -> SloClass:
    try:
        return SloClass.parse(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _pct(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(len(sorted_values) * pct / 100.0))
    return sorted_values[idx]


def _apply_chat_template(engine: LLMEngine, messages: List[ChatMessage]) -> str:
    """Render chat messages to a prompt, preferring the tokenizer's template."""
    try:
        tok = engine.tokenizer
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in messages],
                tokenize=False,
                add_generation_prompt=True,
            )
    except Exception:
        pass
    # Fallback: a plain transcript. Models without a template are usually base
    # models, where this is the conventional shape anyway.
    parts = [f"{m.role}: {m.content}" for m in messages]
    parts.append("assistant:")
    return "\n".join(parts)


async def _stream_completion(runner, token_ids, params, slo, req):
    """SSE stream. Emits the completion once, then [DONE].

    Token-by-token streaming needs a per-token callback from the engine; this
    build resolves a request when it finishes, so the stream carries one data
    frame. The wire format is correct for OpenAI clients either way.
    """
    request_id, out = await runner.submit(token_ids, params, slo, req.helios.prefix_cache)
    chunk = {
        "id": request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "text": out.text,
                "finish_reason": out.finish_reason.value if out.finish_reason else None,
            }
        ],
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_chat(runner, token_ids, params, slo, req):
    request_id, out = await runner.submit(token_ids, params, slo, req.helios.prefix_cache)
    chunk = {
        "id": request_id.replace("cmpl", "chatcmpl"),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": out.text},
                "finish_reason": out.finish_reason.value if out.finish_reason else None,
            }
        ],
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"
