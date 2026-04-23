"""8.a — Embedder Protocol + FakeEmbedder."""
import pytest

from accord_ai.knowledge.embedder import Embedder
from accord_ai.knowledge.fake_embedder import FakeEmbedder


# --- Shape ---

def test_default_dimension_is_384():
    """384 matches all-MiniLM-L6-v2, the expected Phase 7 real impl."""
    assert FakeEmbedder().dimension == 384


def test_dimension_is_configurable():
    assert FakeEmbedder(dimension=128).dimension == 128


# --- embed() ---

@pytest.mark.asyncio
async def test_embed_returns_vector_of_configured_dimension():
    e = FakeEmbedder(dimension=64)
    vec = await e.embed("some text")
    assert len(vec) == 64


@pytest.mark.asyncio
async def test_embed_returns_floats():
    vec = await FakeEmbedder(dimension=8).embed("hello")
    assert all(isinstance(x, float) for x in vec)


@pytest.mark.asyncio
async def test_same_input_produces_same_vector_across_instances():
    """Deterministic hash-seeding — independent of instance identity."""
    v1 = await FakeEmbedder().embed("hello world")
    v2 = await FakeEmbedder().embed("hello world")
    assert v1 == v2


@pytest.mark.asyncio
async def test_different_inputs_produce_different_vectors():
    e = FakeEmbedder()
    assert await e.embed("hello") != await e.embed("world")


# --- embed_batch() ---

@pytest.mark.asyncio
async def test_embed_batch_returns_list_per_input_with_correct_dim():
    vecs = await FakeEmbedder(dimension=32).embed_batch(["a", "b", "c"])
    assert len(vecs) == 3
    assert all(len(v) == 32 for v in vecs)


@pytest.mark.asyncio
async def test_embed_batch_matches_embed_loop():
    """Batch must equal loop(embed) — same vectors in same order."""
    e1 = FakeEmbedder(dimension=16)
    e2 = FakeEmbedder(dimension=16)
    texts = ["one", "two", "three"]
    batched = await e1.embed_batch(texts)
    one_by_one = [await e2.embed(t) for t in texts]
    assert batched == one_by_one


@pytest.mark.asyncio
async def test_empty_batch_returns_empty_list():
    assert await FakeEmbedder().embed_batch([]) == []


# --- Call tracking ---

@pytest.mark.asyncio
async def test_calls_records_input_history_in_order():
    e = FakeEmbedder()
    await e.embed("first")
    await e.embed_batch(["second", "third"])
    assert e.calls == ["first", "second", "third"]


# --- Protocol conformance ---

@pytest.mark.asyncio
async def test_conforms_to_embedder_protocol():
    """FakeEmbedder satisfies the Embedder Protocol structurally."""
    embedder: Embedder = FakeEmbedder()
    vec = await embedder.embed("q")
    assert len(vec) == embedder.dimension
