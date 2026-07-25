"""Token sampling.

Determinism note: greedy decoding (temperature == 0) must be exactly
reproducible, and it is the mode every parity test uses. Stochastic sampling
takes a per-request seeded generator rather than the global torch RNG so that
concurrent requests cannot perturb each other's token streams -- with a shared
global generator, the tokens a request receives would depend on how many other
requests happened to be in the batch, which makes bugs unreproducible.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ..core.types import SamplingParams


class Sampler:
    """Applies temperature, top-k, and top-p, then draws a token."""

    def __init__(self) -> None:
        # One generator per request id, created lazily.
        self._generators: Dict[str, torch.Generator] = {}

    def _generator(self, request_id: str, seed: Optional[int]) -> Optional[torch.Generator]:
        if seed is None:
            return None
        gen = self._generators.get(request_id)
        if gen is None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)
            self._generators[request_id] = gen
        return gen

    def release(self, request_id: str) -> None:
        self._generators.pop(request_id, None)

    def sample(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        request_id: str = "",
    ) -> int:
        """Draw one token from a [vocab_size] logits vector."""
        logits = logits.squeeze()
        if logits.dim() != 1:
            raise ValueError(f"expected 1-D logits, got shape {tuple(logits.shape)}")

        if params.greedy:
            return int(torch.argmax(logits).item())

        logits = logits.float() / params.temperature

        if params.top_k > 0:
            k = min(params.top_k, logits.numel())
            kth = torch.topk(logits, k).values[-1]
            logits = torch.where(
                logits < kth, torch.full_like(logits, float("-inf")), logits
            )

        if params.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            # Keep the smallest prefix whose mass exceeds top_p. The shift keeps
            # the token that crosses the threshold, so top_p never yields an
            # empty candidate set.
            cutoff = cumulative > params.top_p
            cutoff = torch.cat(
                [torch.zeros(1, dtype=torch.bool, device=cutoff.device), cutoff[:-1]]
            )
            sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(
                0, sorted_idx, sorted_logits
            )

        probs = torch.softmax(logits, dim=-1)
        gen = self._generator(request_id, params.seed)
        return int(torch.multinomial(probs, num_samples=1, generator=gen).item())

    def logprobs(
        self, logits: torch.Tensor, top_n: int, chosen: int
    ) -> Dict[int, float]:
        """Top-n log probabilities plus the chosen token's, for the API response."""
        logprobs = torch.log_softmax(logits.squeeze().float(), dim=-1)
        top = torch.topk(logprobs, min(top_n, logprobs.numel()))
        out = {int(i.item()): float(v.item()) for v, i in zip(top.values, top.indices)}
        out[chosen] = float(logprobs[chosen].item())
        return out


def greedy_token(logits: torch.Tensor) -> int:
    """Argmax helper used by the speculative draft path and parity tests."""
    return int(torch.argmax(logits.squeeze()).item())
