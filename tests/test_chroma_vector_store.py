"""8.d — ChromaVectorStore. Uses real chromadb with tmp_path-backed store
(fast enough to always run — PersistentClient init is <100ms)."""
import pytest

from accord_ai.knowledge.chroma_vector_store import ChromaVectorStore
from accord_ai.knowledge.vector_store import DimensionMismatchError


@pytest.fixture
def store(tmp_path):
    return ChromaVectorStore(
        path=str(tmp_path / "chroma"),
        collection_name="test",
        dimension=4,
    )


# --- Round-trip ---

@pytest.mark.asyncio
async def test_add_and_query_roundtrip(store):
    await store.add(
        doc_id="doc1",
        embedding=[0.1, 0.2, 0.3, 0.4],
        document="first document",
        metadata={"lob": "ca"},
    )
    hits = await store.query(embedding=[0.1, 0.2, 0.3, 0.4])
    assert len(hits) == 1
    assert hits[0].doc_id == "doc1"
    assert hits[0].document == "first document"
    assert hits[0].metadata == {"lob": "ca"}
    assert hits[0].distance >= 0.0


@pytest.mark.asyncio
async def test_query_k_limit_respected(store):
    for i in range(5):
        await store.add(
            doc_id=f"d{i}",
            embedding=[float(i), 0.0, 0.0, 0.0],
            document=f"doc {i}",
        )
    hits = await store.query(embedding=[0.0, 0.0, 0.0, 0.0], k=3)
    assert len(hits) == 3


@pytest.mark.asyncio
async def test_query_returns_sorted_by_distance_ascending(store):
    await store.add(doc_id="near",   embedding=[1.0, 0.01, 0.0, 0.0], document="near")
    await store.add(doc_id="far",    embedding=[0.0, 1.0, 0.0, 0.0], document="far")
    await store.add(doc_id="middle", embedding=[0.7, 0.7, 0.0, 0.0], document="middle")

    hits = await store.query(embedding=[1.0, 0.0, 0.0, 0.0])
    ids = [h.doc_id for h in hits]
    assert ids[0] == "near"
    assert ids[-1] == "far"


# --- Dimension enforcement ---

@pytest.mark.asyncio
async def test_add_dimension_mismatch_raises(store):
    with pytest.raises(DimensionMismatchError):
        await store.add(doc_id="x", embedding=[0.1, 0.2], document="x")


@pytest.mark.asyncio
async def test_query_dimension_mismatch_raises(store):
    with pytest.raises(DimensionMismatchError):
        await store.query(embedding=[0.1, 0.2])


# --- Upsert + empty metadata + persistence + tenant isolation ---

@pytest.mark.asyncio
async def test_add_overwrites_existing_doc_id(store):
    await store.add(doc_id="x", embedding=[1.0, 0.0, 0.0, 0.0], document="first")
    await store.add(doc_id="x", embedding=[1.0, 0.0, 0.0, 0.0], document="second")
    hits = await store.query(embedding=[1.0, 0.0, 0.0, 0.0])
    matching = [h for h in hits if h.doc_id == "x"]
    assert len(matching) == 1
    assert matching[0].document == "second"


@pytest.mark.asyncio
async def test_empty_metadata_roundtrips_as_empty_dict(store):
    """Placeholder metadata written for None is stripped on read."""
    await store.add(doc_id="x", embedding=[1.0, 0.0, 0.0, 0.0], document="no meta")
    [hit] = await store.query(embedding=[1.0, 0.0, 0.0, 0.0])
    assert hit.metadata == {}


@pytest.mark.asyncio
async def test_persistence_across_instances(tmp_path):
    path = str(tmp_path / "chroma")
    s1 = ChromaVectorStore(path=path, collection_name="persist", dimension=4)
    await s1.add(doc_id="persist", embedding=[1.0, 0.0, 0.0, 0.0], document="saved")

    s2 = ChromaVectorStore(path=path, collection_name="persist", dimension=4)
    hits = await s2.query(embedding=[1.0, 0.0, 0.0, 0.0])
    assert hits[0].doc_id == "persist"


@pytest.mark.asyncio
async def test_collections_isolated_at_same_db_path(tmp_path):
    path = str(tmp_path / "chroma")
    acme   = ChromaVectorStore(path=path, collection_name="acme",   dimension=4)
    globex = ChromaVectorStore(path=path, collection_name="globex", dimension=4)

    await acme.add(doc_id="a", embedding=[1.0, 0.0, 0.0, 0.0], document="acme doc")
    hits = await globex.query(embedding=[1.0, 0.0, 0.0, 0.0])
    assert hits == []


@pytest.mark.asyncio
async def test_metadata_scalar_types_roundtrip(store):
    """chromadb accepts str/int/float/bool in metadata."""
    await store.add(
        doc_id="d",
        embedding=[1.0, 0.0, 0.0, 0.0],
        document="typed",
        metadata={"lob": "ca", "year": 2026, "active": True, "score": 0.95},
    )
    [hit] = await store.query(embedding=[1.0, 0.0, 0.0, 0.0])
    assert hit.metadata["lob"] == "ca"
    assert hit.metadata["year"] == 2026
    assert hit.metadata["active"] is True
    assert hit.metadata["score"] == 0.95


# --- P7.0: sentinel rename + metadata={} behavior pin ---

@pytest.mark.asyncio
async def test_underscore_metadata_key_roundtrips_unharmed(store):
    """Namespaced sentinel leaves the '_' key namespace free for callers."""
    await store.add(
        doc_id="x",
        embedding=[1.0, 0.0, 0.0, 0.0],
        document="temp",
        metadata={"_": "legit-value"},
    )
    [hit] = await store.query(embedding=[1.0, 0.0, 0.0, 0.0])
    assert hit.metadata == {"_": "legit-value"}


@pytest.mark.asyncio
async def test_explicit_empty_metadata_reads_as_empty_dict(store):
    """metadata={} behaves same as None — placeholder written, stripped on read.
    Pins (a): single code path, symmetric roundtrip."""
    await store.add(
        doc_id="x",
        embedding=[1.0, 0.0, 0.0, 0.0],
        document="explicit empty",
        metadata={},
    )
    [hit] = await store.query(embedding=[1.0, 0.0, 0.0, 0.0])
    assert hit.metadata == {}
