"""P9.0 — pin the library's top-level + testing-subpackage public exports."""


def test_top_level_exports():
    import accord_ai
    from accord_ai import IntakeApp, Settings, build_intake_app
    assert callable(build_intake_app)
    assert isinstance(accord_ai.__version__, str)
    assert "IntakeApp" in accord_ai.__all__
    assert "Settings" in accord_ai.__all__
    assert "build_intake_app" in accord_ai.__all__


def test_testing_subpackage_exports():
    from accord_ai.testing import FakeEmbedder, FakeEngine, FakeVectorStore
    assert callable(FakeEngine)
    assert callable(FakeEmbedder)
    assert callable(FakeVectorStore)


def test_testing_subpackage_reexports_match_source():
    """Re-exports must be the same class objects, not aliases."""
    from accord_ai.knowledge.fake_embedder import FakeEmbedder as _E
    from accord_ai.knowledge.fake_vector_store import FakeVectorStore as _V
    from accord_ai.llm.fake_engine import FakeEngine as _Eng
    from accord_ai.testing import FakeEmbedder, FakeEngine, FakeVectorStore
    assert FakeEngine is _Eng
    assert FakeEmbedder is _E
    assert FakeVectorStore is _V
