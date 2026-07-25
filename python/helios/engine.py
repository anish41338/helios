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
    # INT8 paged KV storage with per-token scales (spec section 7.4). Roughly
    # doubles the number of resident sequences at the same byte budget, at a
    # bounded accuracy cost measured in tests/quant/test_kv_quant.py.
    quantize_kv: bool = False
    # QASSD (spec section 7): draft speculation from a 4-bit view of the same
    # weights instead of from the target itself. INCREASES memory (both precisions
    # resident, plus a second KV cache -- see memory_report()), and is
    # only meaningful with enable_spec_decode.
    quantized_draft: bool = False
    quant_group_size: int = 128
    # Prompts used to calibrate AWQ scales for the draft. Without them the draft
    # is plain round-to-nearest, which measurably accepts less often.
    quant_calib_prompts: Optional[List[str]] = None


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

        # Derive block count from the byte budget (spec section 5.2). A quantized
        # cache stores int8 payload plus one fp16 scale per (token, head), and
        # both terms go into the divisor -- otherwise the engine hands out more
        # blocks than the cache can hold.
        kv_dtype_bytes = 1 if config.quantize_kv else self.dtype_bytes
        scale_bytes = 2 if config.quantize_kv else 0
        bytes_per_block = Allocator.bytes_per_block(
            block_size=config.block_size,
            n_kv_heads=self.model_config.num_key_value_heads,
            head_dim=self.model_config.head_dim,
            n_layers=self.model_config.num_hidden_layers,
            dtype_bytes=kv_dtype_bytes,
            scale_bytes_per_token=scale_bytes,
        )
        num_blocks = max(8, config.kv_cache_bytes // bytes_per_block)

        max_len = config.max_model_len or self.model_config.max_position_embeddings
        # A context longer than the KV pool can hold is a promise we cannot
        # keep, so clamp it and let add_request reject anything above.
        max_len = min(max_len, num_blocks * config.block_size)

        self.dual = self._build_dual(model, config)

        self.runner = ModelRunner(
            model=model,
            num_blocks=num_blocks,
            block_size=config.block_size,
            device=config.device,
            dtype=self.dtype,
            kv_quant=config.quantize_kv,
            dual=self.dual,
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
        # request_id -> pending text deltas, for streaming clients. Populated only
        # for requests that called open_stream, so a non-streaming workload pays
        # nothing (not even the incremental detokenize).
        self._stream_queues: Dict[str, List[str]] = {}
        self._stream_sent: Dict[str, int] = {}

    def _build_dual(self, model, config: EngineConfig):
        """Build the int4 draft view, if QASSD is enabled.

        Requires speculation to be on: an int4 copy that nothing drafts from is
        pure memory overhead, so asking for one without speculation is a config
        error rather than something to silently ignore.
        """
        if not config.quantized_draft:
            return None
        if not config.enable_spec_decode:
            raise ValueError(
                "quantized_draft=True requires enable_spec_decode=True. The int4 "
                "draft is only read by the speculative path, so without it the copy "
                "is pure overhead -- and it is an increase, not a saving: both "
                "precisions stay resident, plus a second KV cache for the draft. "
                "Call engine.memory_report() for the exact figures."
            )

        from .exec.qassd import DualPrecisionModel
        from .exec.quant import QuantConfig

        calib = None
        if config.quant_calib_prompts:
            calib = [self.tokenizer.encode(p) for p in config.quant_calib_prompts]
        return DualPrecisionModel(
            model,
            QuantConfig(group_size=config.quant_group_size),
            calib_token_ids=calib,
        )

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
        self._emit_deltas(outputs)

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

    # --------------------------------------------------------------- streaming

    def _emit_deltas(self, outputs) -> None:
        """Queue incremental text for every sequence that produced tokens.

        Streaming is built on a per-request delta queue drained by the frontend,
        rather than a callback, because the engine is single-threaded and the
        frontend is async: a callback would run arbitrary user code inside the
        scheduler step, which is exactly what the determinism contract forbids
        (spec section 19.4). A queue keeps the seam.

        Detokenization is incremental and this is the subtle part. A BPE token is
        not a character, so decoding each token id alone produces mojibake for
        anything multi-byte -- an emoji or a CJK character spans several tokens.
        Decoding the whole output every step and taking the new suffix is correct
        by construction: the tokenizer sees the full context it needs, and a
        partial character simply does not appear in the decoded string until its
        last token arrives. Cost is O(output_len) per step, which is why the
        decoded prefix length is cached rather than the string re-diffed.
        """
        if not self._stream_queues:
            return
        for out in outputs.outputs:
            if not out.token_ids:
                continue
            rid = self._seq_to_request.get(out.seq_id)
            queue = self._stream_queues.get(rid) if rid else None
            if queue is None:
                continue
            seq = self.scheduler.get_sequence(out.seq_id)
            if seq is None:
                continue
            try:
                text = self.tokenizer.decode(seq.output_token_ids)
            except Exception:
                continue
            sent = self._stream_sent.get(rid, 0)
            if len(text) > sent:
                queue.append(text[sent:])
                self._stream_sent[rid] = len(text)

    def open_stream(self, request_id: str) -> List[str]:
        """Register `request_id` for incremental deltas. Returns its queue."""
        queue: List[str] = []
        self._stream_queues[request_id] = queue
        self._stream_sent[request_id] = 0
        return queue

    def close_stream(self, request_id: str) -> None:
        self._stream_queues.pop(request_id, None)
        self._stream_sent.pop(request_id, None)

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

    def memory_report(self) -> Dict[str, float]:
        """Complete resident-memory accounting for this engine.

        Exists because `DualPrecisionModel.memory_overhead()` covers only
        *weights*, and quoting it as "the cost of QASSD" understates the real
        figure: enabling a quantized draft also allocates an entire second KV
        cache (the section 7.4 shadow), because the draft's keys and values come
        from int4 weights and are numerically not the target's. On this engine
        that shadow is ~0.28x the main cache -- not nothing, and not visible
        anywhere in the weight-only number.

        Reported as one dict so the two costs cannot be quoted separately by
        accident.
        """
        from .exec.quant import QuantLinear

        target_w = sum(
            p.numel() * p.element_size() for p in self.runner.model.parameters()
        )
        main_kv = self.runner.kv_bytes
        draft_w = 0
        draft_kv = 0
        if self.dual is not None:
            draft_w = sum(
                p.numel() * p.element_size() for p in self.dual.draft.parameters()
            ) + sum(
                m.resident_bytes()
                for m in self.dual.draft.modules()
                if isinstance(m, QuantLinear)
            )
        if self.runner.draft_kv_caches:
            draft_kv = sum(c.nbytes for c in self.runner.draft_kv_caches)

        base = target_w + main_kv
        total = base + draft_w + draft_kv
        return {
            "target_weight_bytes": target_w,
            "main_kv_bytes": main_kv,
            "draft_weight_bytes": draft_w,
            "draft_kv_bytes": draft_kv,
            "baseline_bytes": base,
            "total_bytes": total,
            # The number to quote for "what does enabling QASSD cost".
            "total_overhead_ratio": total / max(1, base),
            "weight_overhead_ratio": (target_w + draft_w) / max(1, target_w),
            "kv_quantized": bool(self.config.quantize_kv),
        }

    def stats_snapshot(self) -> Dict[str, object]:
        """Values backing the Prometheus surface (spec section 9.2)."""
        stats = self.scheduler.stats
        extra: Dict[str, object] = {}
        if self.dual is not None:
            # The measured acceptance rate, separate from the scheduler's sliding
            # window: with a quantized draft this is a real quantity that operators
            # need to watch, because a drop in it means speculation has started
            # costing throughput rather than saving it.
            extra["helios_qassd_alpha"] = self.runner.measured_acceptance
            extra["helios_qassd_drafted_total"] = self.runner.spec_drafted
            extra["helios_qassd_accepted_total"] = self.runner.spec_accepted
            extra["helios_qassd_weight_overhead_ratio"] = self.dual.memory_overhead()[
                "overhead_ratio"
            ]
        extra["helios_kv_quantized"] = int(self.config.quantize_kv)
        extra["helios_kv_bytes_total"] = self.runner.kv_bytes
        return {**extra, **{
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
        }}
