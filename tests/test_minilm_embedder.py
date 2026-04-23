"""8.d — MiniLMEmbedder. Unit tests stub SentenceTransformer; integration
tests (gated) load the real model."""
import os
from unittest.mock import MagicMock

import numpy as np
import pytest

from accord_ai.knowledge.minilm_embedder import MiniLMEmbedder


def _fake_model(dim=384):
    model = MagicMock()
    model.get_sentence_embedding_dimension = lambda: dim

    def encode(texts, convert_to_numpy=True, show_progress_bar=False):
        # Deterministic fake vector per text
        return np.array(
            [[float(abs(hash(t)) % 1000) / 1000.0] * dim for t in texts]
        )

    model.encode = encode
    return model


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(
        "accord_ai.knowledge.minilm_embedder.SentenceTransformer",
        lambda name: _fake_model(),
    )
    return MiniLMEmbedder()


# --- Unit (stubbed model) ---

@pytest.mark.asyncio
async def test_embed_returns_dimension_length_vector(stubbed):
    vec = await stubbed.embed("hello")
    assert len(vec) == 384


@pytest.mark.asyncio
async def test_embed_batch_returns_list_per_input(stubbed):
    vecs = await stubbed.embed_batch(["a", "b", "c"])
    assert len(vecs) == 3
    assert all(len(v) == 384 for v in vecs)


@pytest.mark.asyncio
async def test_embed_batch_empty_skips_model_call(monkeypatch):
    call_count = [0]

    def tracking_factory(name):
        call_count[0] += 1
        return _fake_model()

    monkeypatch.setattr(
        "accord_ai.knowledge.minilm_embedder.SentenceTransformer",
        tracking_factory,
    )
    embedder = MiniLMEmbedder()
    result = await embedder.embed_batch([])
    assert result == []
    assert call_count[0] == 0   # model never loaded


@pytest.mark.asyncio
async def test_same_input_same_vector(stubbed):
    assert await stubbed.embed("hello") == await stubbed.embed("hello")


@pytest.mark.asyncio
async def test_model_lazy_loaded_once(monkeypatch):
    load_count = [0]

    def tracking_factory(name):
        load_count[0] += 1
        return _fake_model()

    monkeypatch.setattr(
        "accord_ai.knowledge.minilm_embedder.SentenceTransformer",
        tracking_factory,
    )
    embedder = MiniLMEmbedder()
    assert load_count[0] == 0   # not loaded at construction
    await embedder.embed("x")
    assert load_count[0] == 1
    await embedder.embed("y")
    assert load_count[0] == 1   # cached


# --- Integration (real MiniLM, gated) ---

@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ACCORD_LLM_INTEGRATION"),
    reason="set ACCORD_LLM_INTEGRATION=1 to run",
)
@pytest.mark.asyncio
async def test_real_minilm_produces_384d_vectors():
    embedder = MiniLMEmbedder()
    vec = await embedder.embed("this is a test sentence")
    assert len(vec) == 384
    assert all(isinstance(x, float) for x in vec)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ACCORD_LLM_INTEGRATION"),
    reason="set ACCORD_LLM_INTEGRATION=1 to run",
)
@pytest.mark.asyncio
async def test_real_minilm_similarity_makes_semantic_sense():
    """Similar sentences should produce higher cosine similarity than dissimilar."""
    embedder = MiniLMEmbedder()
    vecs = await embedder.embed_batch([
        "I love my dog",
        "My dog is wonderful",
        "Quantum field theory is complex",
    ])

    def cos_sim(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb)

    near = cos_sim(vecs[0], vecs[1])
    far = cos_sim(vecs[0], vecs[2])
    assert near > far


# --- Declared-vs-actual dimension enforcement (P7.0) ---

@pytest.mark.asyncio
async def test_declared_dim_must_match_model_output(monkeypatch):
    """Constructor accepts any declared dim; first-load check rejects mismatch."""
    monkeypatch.setattr(
        "accord_ai.knowledge.minilm_embedder.SentenceTransformer",
        lambda name: _fake_model(dim=768),
    )
    embedder = MiniLMEmbedder(model_name="fake", dimension=384)
    with pytest.raises(ValueError, match="768"):
        await embedder.embed("x")
