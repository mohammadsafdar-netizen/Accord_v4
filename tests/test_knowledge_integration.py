"""8.d integration — full real stack end-to-end.

Gated by ACCORD_LLM_INTEGRATION=1. First run downloads MiniLM (~80MB).
"""
import os

import pytest

from accord_ai.knowledge import Retriever
from accord_ai.knowledge.chroma_vector_store import ChromaVectorStore
from accord_ai.knowledge.minilm_embedder import MiniLMEmbedder


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("ACCORD_LLM_INTEGRATION"),
        reason="set ACCORD_LLM_INTEGRATION=1 to run",
    ),
]


@pytest.mark.asyncio
async def test_real_minilm_plus_real_chroma_end_to_end(tmp_path):
    """Real MiniLM + real Chroma + Retriever. The production shape."""
    retriever = Retriever(
        embedder=MiniLMEmbedder(),
        store=ChromaVectorStore(
            path=str(tmp_path / "chroma"),
            collection_name="e2e",
            dimension=384,
        ),
    )

    docs = [
        ("class_91580", "Class code 91580 covers clerical office employees",
         {"type": "class_code"}),
        ("class_8810", "Class code 8810 is general clerical workers",
         {"type": "class_code"}),
        ("naics_484", "NAICS 484 covers truck transportation and trucking industry",
         {"type": "naics"}),
    ]
    for doc_id, text, metadata in docs:
        await retriever.add_document(doc_id=doc_id, document=text, metadata=metadata)

    # Trucking query → naics_484 should rank first
    hits = await retriever.retrieve("trucking company classification code")
    assert hits[0].doc_id == "naics_484"
