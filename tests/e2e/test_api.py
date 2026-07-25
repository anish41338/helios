"""End-to-end HTTP API tests.

Spec sections 9.1 (OpenAI compatibility) and 13.4 (fuzzing malformed requests).
"""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest
from fastapi.testclient import TestClient

from helios.api.server import create_app
from helios.engine import EngineConfig
from helios.exec.loader import save_toy_model
from helios.exec.model import ModelConfig

TOY_CONFIG = ModelConfig(
    vocab_size=259,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=512,
)


@pytest.fixture(scope="module")
def client():
    path = os.path.join(tempfile.gettempdir(), "helios_api_toy")
    shutil.rmtree(path, ignore_errors=True)
    save_toy_model(path, TOY_CONFIG, seed=0)
    app = create_app(
        EngineConfig(
            model_dir=path,
            kv_cache_bytes=8 * 1024 * 1024,
            block_size=16,
            max_model_len=256,
        )
    )
    with TestClient(app) as c:
        yield c
    shutil.rmtree(path, ignore_errors=True)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["kv_blocks"] > 0


def test_models_lists_helios(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "helios"


def test_metrics_is_prometheus_text(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "helios_kv_blocks_total" in r.text
    # Every line must be `name value`, or Prometheus will reject the scrape.
    for line in r.text.strip().splitlines():
        parts = line.split()
        assert len(parts) == 2, f"malformed metric line: {line!r}"
        float(parts[1])


def test_completions_returns_usage_and_choice(client):
    r = client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert len(body["choices"]) == 1
    assert body["usage"]["completion_tokens"] == 5
    assert body["choices"][0]["finish_reason"] == "length"


def test_chat_completions_returns_assistant_message(client):
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"]["completion_tokens"] == 4


def test_helios_extension_block_is_accepted(client):
    r = client.post(
        "/v1/completions",
        json={
            "prompt": "hello",
            "max_tokens": 3,
            "helios": {"slo_class": "A", "prefix_cache": False, "deadline_ms": 500},
        },
    )
    assert r.status_code == 200


def test_streaming_returns_sse_frames(client):
    r = client.post(
        "/v1/completions", json={"prompt": "hi", "max_tokens": 3, "stream": True}
    )
    assert r.status_code == 200
    assert "data: " in r.text
    assert "[DONE]" in r.text


def test_greedy_requests_are_reproducible(client):
    payload = {"prompt": "same prompt", "max_tokens": 6, "temperature": 0.0}
    first = client.post("/v1/completions", json=payload).json()
    second = client.post("/v1/completions", json=payload).json()
    assert first["choices"][0]["text"] == second["choices"][0]["text"]


# ------------------------------------------------------------------ fuzzing


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": "", "max_tokens": 4},                    # empty prompt
        {"prompt": "x", "max_tokens": 0},                   # non-positive length
        {"prompt": "x", "max_tokens": -5},
        {"prompt": "x", "max_tokens": 4, "temperature": -1.0},
        {"prompt": "x", "max_tokens": 4, "top_p": 0.0},
        {"prompt": "x", "max_tokens": 4, "top_p": 1.5},
        {"prompt": "x", "max_tokens": 4, "top_k": -2},
        {"prompt": "x", "max_tokens": 4, "logprobs": 99},
        {"prompt": "x", "max_tokens": 999999},              # exceeds context
        {"prompt": "x", "max_tokens": 4, "helios": {"slo_class": "Q"}},
    ],
)
def test_malformed_requests_are_rejected_not_crashed(client, payload):
    """Spec section 13.4: bad input must produce a 4xx, never a 500."""
    r = client.post("/v1/completions", json=payload)
    assert 400 <= r.status_code < 500, f"expected client error, got {r.status_code}"


def test_unicode_prompt_round_trips(client):
    r = client.post(
        "/v1/completions",
        json={"prompt": "héllo 世界 🌞 \u200b edge", "max_tokens": 3},
    )
    assert r.status_code == 200


def test_empty_messages_rejected(client):
    r = client.post("/v1/chat/completions", json={"messages": [], "max_tokens": 4})
    assert r.status_code == 400


def test_concurrent_requests_all_complete(client):
    """Several in-flight requests must all be served by the single-threaded engine."""
    import concurrent.futures

    def one(i: int):
        return client.post(
            "/v1/completions", json={"prompt": f"p{i}", "max_tokens": 4}
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(one, range(6)))

    assert all(r.status_code == 200 for r in responses)
    assert all(r.json()["usage"]["completion_tokens"] == 4 for r in responses)
