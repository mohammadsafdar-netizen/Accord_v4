"""8.c — Retriever composition + dim-safety + public surface."""
import pytest

from accord_ai.knowledge import (
    DimensionMismatchError,
    FakeEmbedder,
    FakeVectorStore,
    Hit,
    Retriever,
)


def _hit(doc_id="d", document="text", distance=0.1, **metadata):
    return Hit(doc_id=doc_id, document=document, metadata=metadata, distance=distance)


# --- Construction dim-safety ---

def test_matching_dimensions_constructs_cleanly():
    Retriever(FakeEmbedder(dimension=4), FakeVectorStore(dimension=4))


def test_mismatched_dimensions_raise_at_construction():
    """Wiring error surfaces BEFORE any retrieval call."""
    with pytest.raises(DimensionMismatchError, match="embedder dim 8 != store dim 4"):
        Retriever(FakeEmbedder(dimension=8), FakeVectorStore(dimension=4))


# --- retrieve() ---

@pytest.mark.asyncio
async def test_retrieve_embeds_query_and_queries_store():
    embedder = FakeEmbedder(dimension=4)
    store = FakeVectorStore(
        results=[[_hit(doc_id="first"), _hit(doc_id="second")]],
        dimension=4,
    )
    retriever = Retriever(embedder, store)

    hits = await retriever.retrieve("hello")

    assert [h.doc_id for h in hits] == ["first", "second"]
    assert embedder.calls == ["hello"]
    # Store received the same vector FakeEmbedder produced for "hello"
    expected_vec = await FakeEmbedder(dimension=4).embed("hello")
    assert store.queries == [expected_vec]


@pytest.mark.asyncio
async def test_retrieve_passes_k_through():
    store = FakeVectorStore(
        results=[[_hit(doc_id=f"d{i}") for i in range(10)]],
        dimension=4,
    )
    retriever = Retriever(FakeEmbedder(dimension=4), store)
    hits = await retriever.retrieve("q", k=3)
    assert len(hits) == 3


@pytest.mark.asyncio
async def test_retrieve_returns_empty_when_store_has_no_results():
    retriever = Retriever(FakeEmbedder(dimension=4), FakeVectorStore(dimension=4))
    assert await retriever.retrieve("q") == []


# --- add_document() ---

@pytest.mark.asyncio
async def test_add_document_embeds_and_stores_with_metadata():
    embedder = FakeEmbedder(dimension=4)
    store = FakeVectorStore(dimension=4)
    retriever = Retriever(embedder, store)

    await retriever.add_document(
        doc_id="doc1",
        document="this is the doc",
        metadata={"lob": "commercial_auto"},
    )

    assert embedder.calls == ["this is the doc"]
    assert store.adds == [{
        "doc_id": "doc1",
        "document": "this is the doc",
        "metadata": {"lob": "commercial_auto"},
    }]


@pytest.mark.asyncio
async def test_add_document_defaults_metadata_to_empty():
    store = FakeVectorStore(dimension=4)
    retriever = Retriever(FakeEmbedder(dimension=4), store)
    await retriever.add_document(doc_id="x", document="text")
    assert store.adds[0]["metadata"] == {}


# --- End-to-end via full fake stack ---

@pytest.mark.asyncio
async def test_end_to_end_add_then_retrieve_uses_both_fakes():
    """Loader-and-query flow. Add path exercised. Retrieve returns the canned hits."""
    embedder = FakeEmbedder(dimension=4)
    store = FakeVectorStore(
        results=[[_hit(doc_id="doc1", document="stored text", distance=0.05)]],
        dimension=4,
    )
    retriever = Retriever(embedder, store)

    await retriever.add_document(doc_id="doc1", document="stored text")
    hits = await retriever.retrieve("query for stored text")

    assert hits[0].doc_id == "doc1"
    assert hits[0].document == "stored text"
    # Both paths exercised the embedder
    assert embedder.calls == ["stored text", "query for stored text"]


# --- Public surface ---

def test_public_surface_is_importable_from_package_root():
    from accord_ai.knowledge import (
        DimensionMismatchError,
        Embedder,
        FakeEmbedder,
        FakeVectorStore,
        Hit,
        Retriever,
        VectorStore,
    )
    assert callable(FakeEmbedder)
    assert callable(FakeVectorStore)
    assert callable(Retriever)
