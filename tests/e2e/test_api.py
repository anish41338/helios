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


def _sse_frames(text):
    """Parse an SSE body into decoded JSON payloads, excluding the [DONE] marker."""
    import json

    out = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        body = line[len("data: "):].strip()
        if body == "[DONE]":
            continue
        out.append(json.loads(body))
    return out


def test_streaming_returns_sse_frames(client):
    r = client.post(
        "/v1/completions", json={"prompt": "hi", "max_tokens": 3, "stream": True}
    )
    assert r.status_code == 200
    assert "data: " in r.text
    assert "[DONE]" in r.text
    assert r.headers["content-type"].startswith("text/event-stream")


def test_streaming_emits_more_than_one_content_frame(client):
    """Incremental streaming, not one frame with the whole answer.

    This is the property the previous build did not have: the wire format was
    correct but every response arrived as a single frame, so a client saw nothing
    until the request finished. `> 1` rather than `== max_tokens` because a text
    delta is not a token -- a step can produce no printable characters, and
    speculative decoding can commit several tokens at once.
    """
    r = client.post(
        "/v1/completions",
        json={"prompt": "streaming test", "max_tokens": 16, "stream": True},
    )
    assert r.status_code == 200
    frames = _sse_frames(r.text)
    content = [f for f in frames if f["choices"][0]["text"]]
    assert len(content) > 1, f"only {len(content)} content frames: {r.text[:400]}"


def test_streamed_text_reassembles_to_the_non_streamed_answer(client):
    """The correctness property: streaming must not change the output.

    Concatenating the deltas must reproduce exactly what the non-streaming
    endpoint returns for the same greedy request. An incremental detokenizer that
    drops or duplicates a character would pass every other test here.
    """
    payload = {"prompt": "reassembly check", "max_tokens": 12, "temperature": 0.0}
    whole = client.post("/v1/completions", json=payload).json()["choices"][0]["text"]

    r = client.post("/v1/completions", json={**payload, "stream": True})
    streamed = "".join(f["choices"][0]["text"] for f in _sse_frames(r.text))
    assert streamed == whole


def test_stream_terminates_with_a_finish_reason(client):
    """OpenAI clients read the finish reason off the last frame before [DONE]."""
    r = client.post(
        "/v1/completions", json={"prompt": "hi", "max_tokens": 4, "stream": True}
    )
    frames = _sse_frames(r.text)
    assert frames[-1]["choices"][0]["finish_reason"] is not None
    assert frames[-1]["choices"][0]["text"] == ""
    assert r.text.rstrip().endswith("data: [DONE]")


def test_chat_stream_sends_the_role_once_then_content(client):
    """The chat.completion.chunk protocol: role in the first delta, then content."""
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello there"}],
            "max_tokens": 12,
            "stream": True,
        },
    )
    assert r.status_code == 200
    frames = _sse_frames(r.text)
    assert all(f["object"] == "chat.completion.chunk" for f in frames)

    roles = [f for f in frames if "role" in f["choices"][0]["delta"]]
    assert len(roles) == 1, "the role must appear exactly once"
    assert roles[0]["choices"][0]["delta"]["role"] == "assistant"
    assert frames.index(roles[0]) == 0, "the role frame must come first"

    content = [f for f in frames if f["choices"][0]["delta"].get("content")]
    assert len(content) > 1
    assert frames[-1]["choices"][0]["finish_reason"] is not None


def test_chat_stream_reassembles_to_the_non_streamed_message(client):
    payload = {
        "messages": [{"role": "user", "content": "chat reassembly"}],
        "max_tokens": 12,
        "temperature": 0.0,
    }
    whole = client.post("/v1/chat/completions", json=payload).json()
    expected = whole["choices"][0]["message"]["content"]

    r = client.post("/v1/chat/completions", json={**payload, "stream": True})
    streamed = "".join(
        f["choices"][0]["delta"].get("content", "") for f in _sse_frames(r.text)
    )
    assert streamed == expected


def test_streaming_does_not_leak_queues(client):
    """A disconnected or finished stream must deregister itself.

    Otherwise every streamed request permanently adds work to each engine step:
    a slow leak that shows up as gradual throughput decay, which is far harder to
    attribute than a crash.
    """
    engine = client.app.state.engine
    for i in range(3):
        client.post(
            "/v1/completions",
            json={"prompt": f"leak check {i}", "max_tokens": 4, "stream": True},
        )
    assert engine._stream_queues == {}
    assert engine._stream_sent == {}


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
