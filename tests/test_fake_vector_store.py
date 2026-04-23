"""8.b — VectorStore Protocol + FakeVectorStore."""
from dataclasses import FrozenInstanceError

import pytest

from accord_ai.knowledge.fake_vector_store import FakeVectorStore
from accord_ai.knowledge.vector_store import (
    DimensionMismatchError,
    Hit,
    VectorStore,
)


def _h(doc_id="doc", document="text", distance=0.1, **metadata):
    return Hit(doc_id=doc_id, document=document, metadata=metadata, distance=distance)


# --- Hit shape ---

def test_hit_is_frozen():
    h = Hit(doc_id="x", document="y", metadata={}, distance=0.1)
    with pytest.raises(FrozenInstanceError):
        h.distance = 0.2


# --- Dimension ---

def test_default_dimension_is_384():
    assert FakeVectorStore().dimension == 384


def test_dimension_is_configurable():
    assert FakeVectorStore(dimension=128).dimension == 128


# --- add() ---

@pytest.mark.asyncio
async def test_add_records_on_adds():
    store = FakeVectorStore(dimension=4)
    await store.add(
        doc_id="a",
        embedding=[0.1, 0.2, 0.3, 0.4],
        document="hello",
        metadata={"lob": "ca"},
    )
    assert store.adds == [
        {"doc_id": "a", "document": "hello", "metadata": {"lob": "ca"}},
    ]


@pytest.mark.asyncio
async def test_add_with_no_metadata_defaults_to_empty_dict():
    store = FakeVectorStore(dimension=4)
    await store.add(doc_id="a", embedding=[0.1] * 4, document="x")
    assert store.adds[0]["metadata"] == {}


@pytest.mark.asyncio
async def test_add_dimension_mismatch_raises():
    store = FakeVectorStore(dimension=4)
    with pytest.raises(DimensionMismatchError):
        await store.add(doc_id="a", embedding=[0.1, 0.2], document="x")


# --- query() ---

@pytest.mark.asyncio
async def test_query_returns_canned_results_fifo():
    store = FakeVectorStore(
        results=[
            [_h(doc_id="first", distance=0.1)],
            [_h(doc_id="second", distance=0.2), _h(doc_id="third", distance=0.3)],
        ],
        dimension=4,
    )
    r1 = await store.query(embedding=[0.0] * 4)
    r2 = await store.query(embedding=[0.0] * 4)
    assert [h.doc_id for h in r1] == ["first"]
    assert [h.doc_id for h in r2] == ["second", "third"]


@pytest.mark.asyncio
async def test_query_returns_empty_when_queue_drained():
    store = FakeVectorStore(dimension=4)
    assert await store.query(embedding=[0.0] * 4) == []


@pytest.mark.asyncio
async def test_query_respects_k_limit():
    hits = [_h(doc_id=f"d{i}", distance=float(i)) for i in range(10)]
    store = FakeVectorStore(results=[hits], dimension=4)
    result = await store.query(embedding=[0.0] * 4, k=3)
    assert [h.doc_id for h in result] == ["d0", "d1", "d2"]


@pytest.mark.asyncio
async def test_query_dimension_mismatch_raises():
    store = FakeVectorStore(dimension=4)
    with pytest.raises(DimensionMismatchError):
        await store.query(embedding=[0.1, 0.2])


@pytest.mark.asyncio
async def test_query_records_on_queries_list():
    store = FakeVectorStore(dimension=4)
    await store.query(embedding=[0.1, 0.2, 0.3, 0.4])
    assert store.queries == [[0.1, 0.2, 0.3, 0.4]]


# --- enqueue() ---

@pytest.mark.asyncio
async def test_enqueue_appends_mid_test():
    store = FakeVectorStore(dimension=4)
    store.enqueue([_h(doc_id="late")])
    result = await store.query(embedding=[0.0] * 4)
    assert result[0].doc_id == "late"


# --- Protocol conformance ---

@pytest.mark.asyncio
async def test_conforms_to_vector_store_protocol():
    store: VectorStore = FakeVectorStore(dimension=4)
    await store.add(doc_id="a", embedding=[0.1] * 4, document="x")
    result = await store.query(embedding=[0.2] * 4)
    assert isinstance(result, list)


# --- Error type ---

def test_dimension_mismatch_error_is_valueerror():
    assert issubclass(DimensionMismatchError, ValueError)
