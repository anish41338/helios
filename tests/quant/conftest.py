"""Fixtures for the quantization tests."""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest

from helios.exec.loader import save_toy_model
from helios.exec.model import ModelConfig

# Wider than the parity suite's toy config: hidden_size must exceed the group
# size for grouping to be exercised at all, and a single-group layer would hide
# any per-group scale indexing bug.
QUANT_TOY_CONFIG = ModelConfig(
    vocab_size=259,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=512,
)


@pytest.fixture(scope="session")
def quant_toy_dir():
    path = os.path.join(tempfile.gettempdir(), "helios_quant_toy")
    shutil.rmtree(path, ignore_errors=True)
    save_toy_model(path, QUANT_TOY_CONFIG, seed=7)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def toy_model_factory(quant_toy_dir):
    """Returns a callable producing a FRESH model each call.

    A factory rather than a fixture value because `quantize_model` mutates the
    model in place: a shared instance would let one test's quantization leak into
    the next, and the tests that compare RTN against AWQ need two independent
    copies of the same weights.
    """
    from helios.exec.loader import load_model

    def make():
        return load_model(quant_toy_dir)

    return make
