"""The engine: binds the deterministic scheduler to a real model executor.

This is the top-level object an embedder uses. The frontend (helios.api) wraps
it in HTTP; the benchmark harness drives it directly.

Sizing note: the number of KV blocks is derived from a byte budget using the
spec section 5.2 formula, so the engine's concurrency limit follows from
hardware rather than from a magic constant.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .core.allocator import Allocator
from .core.prefix_cache import PrefixCache
from .core.scheduler import Scheduler, SchedulerConfig
from .core.types import (
    CompletionOutput,
    Request,
    SamplingParams,
    SeqId,
    Sequence,
    SloClass,
)
from .exec.model import ModelConfig
from .exec.runner import ModelRunner


@dataclass
class EngineConfig:
    """Everything needed to stand up an engine."""

    model_dir: str
    kv_cache_bytes: int = 512 * 1024 * 1024   # 512 MiB default KV budget
    block_size: int = 16
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 2048
    max_model_len: Optional[int] = None       # None -> model's max_position_embeddings
    device: str = "cpu"
    enable_prefix_cache: bool = True
    enable_chunked_prefill: bool = True
    enable_spec_decode: bool = False
    spec_gamma: int = 4
    watermark: float = 0.01
    tokenizer_dir: Optional[str] = None


@dataclass
class RequestMetrics:
    """Per-request latency record, in wall-clock seconds.

    TTFT and TPOT are measured here rather than in the scheduler because the
    scheduler is deliberately clock-free (spec section 19.4).
    """

    request_id: str
    arrival_time: float
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    slo_class: SloClass = SloClass.B
    finish_reason: Optional[str] = None

    @property
    def ttft(self) -> Optional[float]:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    @property
    def tpot(self) -> Optional[float]:
        """Mean inter-token latency after the first token."""
        if (
            self.first_token_time is None
            or self.finish_time is None
            or self.output_tokens <= 1
        ):
            return None
        return (self.finish_time - self.first_token_time) / (self.output_tokens - 1)

    @property
    def e2e(self) -> Optional[float]:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time

    def meets_slo(self) -> bool:
        """Goodput accounting: did this request hit its class's targets?"""
        from .core.types import SLO_TARGETS

        targets = SLO_TARGETS[self.slo_class]
        if targets["ttft"] is not None:
            if self.ttft is None or self.ttft > targets["ttft"]:
                return False
        if targets["tpot"] is not None and self.tpot is not None:
            if self.tpot > targets["tpot"]:
                return False
        return True


class LLMEngine:
    """Scheduler + executor + tokenizer, driven by step()."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        model_dir = Path(config.model_dir)

        from .exec.loader import load_config, load_model

        self.model_config: ModelConfig = load_config(model_dir)
        # One dtype decision, used for the weights, the KV cache, and the block
        # sizing below. Deriving them separately is how you get a cache sized for
        # fp32 holding fp16 tensors.
        import torch as _torch

        self.dtype = (
            _torch.float16 if config.device.startswith("cuda") else _torch.float32
        )
        self.dtype_bytes = 2 if self.dtype is _torch.float16 else 4
        model = load_model(model_dir, device=config.device, dtype=self.dtype)

        # Derive block count from the byte budget (spec section 5.2).
        bytes_per_block = Allocator.bytes_per_block(
            block_size=config.block_size,
            n_kv_heads=self.model_config.num_key_value_heads,
            head_dim=self.model_config.head_dim,
            n_layers=self.model_config.num_hidden_layers,
            dtype_bytes=self.dtype_bytes,
        )
        num_blocks = max(8, config.kv_cache_bytes // bytes_per_block)

        max_len = config.max_model_len or self.model_config.max_position_embeddings
        # A context longer than the KV pool can hold is a promise we cannot
        # keep, so clamp it and let add_request reject anything above.
        max_len = min(max_len, num_blocks * config.block_size)

        self.runner = ModelRunner(
            model=model,
            num_blocks=num_blocks,
            block_size=config.block_size,
            device=config.device,
            dtype=self.dtype,
        )

        self.allocator = Allocator(
            total_vram_blocks=num_blocks,
            block_size=config.block_size,
            watermark=config.watermark,
        )
        sched_config = SchedulerConfig(
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            max_model_len=max_len,
            block_size=config.block_size,
            watermark=config.watermark,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_cache=config.enable_prefix_cache,
            enable_spec_decode=config.enable_spec_decode,
            spec_gamma=config.spec_gamma,
        )
        self.scheduler = Scheduler(
            sched_config,
            self.allocator,
            PrefixCache(config.block_size, enabled=config.enable_prefix_cache),
        )

        self.num_blocks = num_blocks
        self.max_model_len = max_len
        self.bytes_per_block = bytes_per_block

        self._tokenizer = None
        self._metrics: Dict[str, RequestMetrics] = {}
        self._seq_to_request: Dict[SeqId, str] = {}
        self._completed: List[CompletionOutput] = []

    # ------------------------------------------------------------- tokenizer

    @property
    def tokenizer(self):
        """Lazily loaded HF tokenizer. Optional: token-id APIs work without it."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            src = self.config.tokenizer_dir or self.config.model_dir
            self._tokenizer = AutoTokenizer.from_pretrained(src)
        return self._tokenizer

    @property
    def eos_token_ids(self) -> List[int]:
        try:
            eos = self.tokenizer.eos_token_id
            return [eos] if eos is not None else []
        except Exception:
            return []

    # --------------------------------------------------------------- submit

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: List[int],
        params: Optional[SamplingParams] = None,
        slo_class: SloClass = SloClass.B,
        prefix_cache: bool = True,
    ) -> SeqId:
        """Queue a request by token ids."""
        params = params or SamplingParams()
        request = Request(
            request_id=request_id,
            prompt_token_ids=list(prompt_token_ids),
            params=params,
            slo_class=slo_class,
            prefix_cache=prefix_cache,
        )
        seq_id = self.scheduler.add_request(request)

        self._seq_to_request[seq_id] = request_id
        self._metrics[request_id] = RequestMetrics(
            request_id=request_id,
            arrival_time=time.perf_counter(),
            prompt_tokens=len(prompt_token_ids),
            slo_class=slo_class,
        )
        return seq_id

    def add_prompt(
        self,
        request_id: str,
        prompt: str,
        params: Optional[SamplingParams] = None,
        slo_class: SloClass = SloClass.B,
    ) -> SeqId:
        """Queue a request by text, tokenizing first."""
        token_ids = self.tokenizer.encode(prompt)
        return self.add_request(request_id, token_ids, params, slo_class)

    def abort(self, request_id: str) -> bool:
        for seq_id, rid in self._seq_to_request.items():
            if rid == request_id:
                return self.scheduler.abort(seq_id)
        return False

    # ----------------------------------------------------------------- step

    def step(self) -> List[CompletionOutput]:
        """Run one scheduler iteration. Returns any requests that completed."""
        outputs = self.scheduler.step(self.runner)
        now = time.perf_counter()

        self._apply_stop_strings(outputs)

        # First-token timing: record when a sequence's first output appears.
        for out in outputs.outputs:
            rid = self._seq_to_request.get(out.seq_id)
            if rid is None or not out.token_ids:
                continue
            metrics = self._metrics.get(rid)
            if metrics and metrics.first_token_time is None:
                metrics.first_token_time = now

        completed: List[CompletionOutput] = []
        for seq in self.scheduler.take_finished():
            rid = self._seq_to_request.get(seq.seq_id, seq.request.request_id)
            metrics = self._metrics.get(rid)
            if metrics:
                metrics.finish_time = now
                metrics.output_tokens = seq.num_output_tokens
                metrics.cached_tokens = seq.cached_prefix_len
                metrics.finish_reason = (
                    seq.finish_reason.value if seq.finish_reason else None
                )
            completed.append(self._make_output(seq))
            self.runner.sampler.release(str(seq.seq_id))

        self._completed.extend(completed)
        return completed

    def _apply_stop_strings(self, outputs) -> None:
        """Terminate sequences whose decoded text contains a stop string.

        Stop *strings* live here rather than in the scheduler because they need
        detokenization, and the scheduler core must stay free of the tokenizer
        (and of anything else that is not a pure function of its own state).
        Stop *token ids* are handled in the scheduler, where they are just an
        integer comparison.

        Without this, a client passing `stop: ["\\n\\n"]` had it silently
        ignored -- the parameter was validated and then never consulted.
        """
        for out in outputs.outputs:
            seq = self.scheduler.get_sequence(out.seq_id)
            if seq is None or seq.is_finished or not out.token_ids:
                continue
            stops = seq.request.params.stop
            if not stops:
                continue

            text = self._decode_cached(seq)
            for s in stops:
                if s and s in text:
                    # Truncate the output at the stop string so the client never
                    # sees it, matching OpenAI's behaviour.
                    cut = text.index(s)
                    seq.output_text_override = text[:cut]
                    self.scheduler.finish_for_stop_string(seq)
                    break

    def _decode_cached(self, seq: Sequence) -> str:
        try:
            return self.tokenizer.decode(seq.output_token_ids)
        except Exception:
            return ""

    def _make_output(self, seq: Sequence) -> CompletionOutput:
        if seq.output_text_override is not None:
            text = seq.output_text_override   # truncated at a stop string
        elif seq.output_token_ids:
            try:
                text = self.tokenizer.decode(seq.output_token_ids)
            except Exception:
                text = ""   # tokenizer unavailable; token ids are still returned
        else:
            text = ""
        return CompletionOutput(
            request_id=seq.request.request_id,
            token_ids=list(seq.output_token_ids),
            text=text,
            finish_reason=seq.finish_reason,
            prompt_tokens=seq.prompt_len,
            cached_tokens=seq.cached_prefix_len,
        )

    def run_until_complete(self, max_steps: int = 100_000) -> List[CompletionOutput]:
        """Drive the engine until every queued request finishes."""
        results: List[CompletionOutput] = []
        steps = 0
        while self.scheduler.has_work and steps < max_steps:
            results.extend(self.step())
            steps += 1
        return results

    def generate(
        self,
        prompts: List[str],
        params: Optional[SamplingParams] = None,
        slo_class: SloClass = SloClass.B,
    ) -> List[CompletionOutput]:
        """Offline batch entry point: submit all prompts, run to completion."""
        for i, prompt in enumerate(prompts):
            self.add_prompt(f"batch-{i}", prompt, params, slo_class)
        results = self.run_until_complete()
        order = {f"batch-{i}": i for i in range(len(prompts))}
        return sorted(results, key=lambda r: order.get(r.request_id, 0))

    # -------------------------------------------------------------- metrics

    def metrics(self) -> List[RequestMetrics]:
        return list(self._metrics.values())

    def stats_snapshot(self) -> Dict[str, object]:
        """Values backing the Prometheus surface (spec section 9.2)."""
        stats = self.scheduler.stats
        return {
            "helios_running_seqs": self.scheduler.num_running(),
            "helios_waiting_seqs": self.scheduler.num_waiting(),
            "helios_kv_blocks_used": self.allocator.committed_vram_blocks,
            "helios_kv_blocks_total": self.allocator.total_vram_blocks,
            "helios_kv_utilization": self.allocator.utilization(),
            "helios_tokens_prefill_total": stats.prefill_tokens,
            "helios_tokens_decode_total": stats.decode_tokens,
            "helios_finished_total": stats.finished_seqs,
            "helios_aborted_total": stats.aborted_seqs,
            "helios_preemptions_recompute_total": stats.preemptions_recompute,
            "helios_preemptions_swap_total": stats.preemptions_swap,
            "helios_spec_acceptance_rate": stats.acceptance_rate,
            "helios_prefix_cache_hit_ratio": self.scheduler.prefix_cache.hit_ratio,
            "helios_exec_faults_total": stats.exec_faults,
            "helios_scheduler_steps_total": stats.step,
            "helios_empty_steps_total": stats.empty_steps,
        }
