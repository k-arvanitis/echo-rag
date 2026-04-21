"""Integration tests for pipeline components — all external services fully mocked."""
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_results(n: int = 3) -> dict:
    """Build a fake ChromaDB query result with n transcript chunks."""
    speakers = ["Alice", "Bob", "Alice"]
    return {
        "documents": [[f"Chunk {i} text." for i in range(n)]],
        "metadatas": [[
            {
                "speaker": speakers[i % len(speakers)],
                "start": float(i),
                "end": float(i + 1),
                "audio_file": "test.mp3",
            }
            for i in range(n)
        ]],
        "distances": [[round(0.1 * (i + 1), 2) for i in range(n)]],
    }


def _make_mock_response(content: str = "The answer.") -> MagicMock:
    """Build a fake OpenAI chat completion response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


# ---------------------------------------------------------------------------
# test_rag_returns_structured_response
# ---------------------------------------------------------------------------

def _mock_encode(embedding: list) -> MagicMock:
    """Return a mock whose .tolist() gives the provided embedding."""
    m = MagicMock()
    m.tolist.return_value = embedding
    return m


def test_rag_returns_structured_response():
    """query_rag returns a (str, list) tuple with a non-empty answer and source chunks."""
    from pipeline.rag import query_rag

    mock_collection = MagicMock()
    mock_embedding_model = MagicMock()
    mock_embedding_model.encode.return_value = _mock_encode([0.0] * 1024)
    mock_collection.query.return_value = _make_mock_results(3)

    with patch("pipeline.rag.HYDE_ENABLED", False), \
         patch("pipeline.rag.RERANKER_ENABLED", False), \
         patch("pipeline.rag._build_openai_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = _make_mock_response("The answer.")

        answer, chunks = query_rag("What did Alice say?", mock_collection, mock_embedding_model)

    assert isinstance(answer, str) and answer
    assert isinstance(chunks, list) and len(chunks) == 3


# ---------------------------------------------------------------------------
# test_empty_retrieval_handled
# ---------------------------------------------------------------------------

def test_empty_retrieval_handled():
    """query_rag returns a clean message and skips the LLM when no chunks are found."""
    from pipeline.rag import query_rag

    mock_collection = MagicMock()
    mock_embedding_model = MagicMock()
    mock_embedding_model.encode.return_value = _mock_encode([0.0] * 1024)
    mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    with patch("pipeline.rag.HYDE_ENABLED", False), \
         patch("pipeline.rag.RERANKER_ENABLED", False), \
         patch("pipeline.rag._build_openai_client") as mock_client:

        answer, chunks = query_rag("What did Alice say?", mock_collection, mock_embedding_model)

    assert "no relevant content" in answer.lower()
    assert chunks == []
    mock_client.return_value.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# test_chroma_unreachable
# ---------------------------------------------------------------------------

def test_chroma_unreachable():
    """query_rag returns a fallback error string when ChromaDB raises, does not crash."""
    from pipeline.rag import query_rag

    mock_collection = MagicMock()
    mock_collection.query.side_effect = Exception("Connection refused")
    mock_embedding_model = MagicMock()
    mock_embedding_model.encode.return_value = _mock_encode([0.0] * 1024)

    with patch("pipeline.rag.HYDE_ENABLED", False), \
         patch("pipeline.rag.RERANKER_ENABLED", False):
        answer, chunks = query_rag("What happened?", mock_collection, mock_embedding_model)

    assert isinstance(answer, str) and len(answer) > 0
    assert chunks == []


# ---------------------------------------------------------------------------
# test_vllm_unreachable
# ---------------------------------------------------------------------------

def test_vllm_unreachable():
    """query_rag returns a fallback error string when the LLM API is unreachable, does not crash."""
    from pipeline.rag import query_rag

    mock_collection = MagicMock()
    mock_embedding_model = MagicMock()
    mock_embedding_model.encode.return_value = _mock_encode([0.0] * 1024)
    mock_collection.query.return_value = _make_mock_results(3)

    with patch("pipeline.rag.HYDE_ENABLED", False), \
         patch("pipeline.rag.RERANKER_ENABLED", False), \
         patch("pipeline.rag.generate_answer", side_effect=RuntimeError("Cannot reach OpenAI API")):
        answer, chunks = query_rag("What happened?", mock_collection, mock_embedding_model)

    assert isinstance(answer, str) and len(answer) > 0


# ---------------------------------------------------------------------------
# test_chunk_boundaries_never_split_speaker_turns
# ---------------------------------------------------------------------------

def test_chunk_boundaries_never_split_speaker_turns():
    """Every chunk boundary aligns with a turn boundary — no turn is split across chunks."""
    from pipeline.chunk import chunk_transcript

    turns = [
        {"speaker": "Alice", "start": 0.0, "end": 1.0, "text": "Hello there, this is a longer sentence from Alice."},
        {"speaker": "Bob",   "start": 1.0, "end": 2.0, "text": "Hi! How are you doing today, Alice?"},
        {"speaker": "Alice", "start": 2.0, "end": 3.0, "text": "I am doing well, thank you for asking Bob."},
        {"speaker": "Bob",   "start": 3.0, "end": 4.0, "text": "Great to hear. Let us get started now."},
    ]
    # max_chars=60 forces splits between turns
    chunks = chunk_transcript(turns, max_chars=60)

    turn_starts = {t["start"] for t in turns}
    turn_ends   = {t["end"]   for t in turns}

    for chunk in chunks:
        assert chunk["start"] in turn_starts, f"chunk start {chunk['start']} is not a turn boundary"
        assert chunk["end"]   in turn_ends,   f"chunk end {chunk['end']} is not a turn boundary"


# ---------------------------------------------------------------------------
# test_speaker_metadata_preserved
# ---------------------------------------------------------------------------

def test_speaker_metadata_preserved():
    """embed_chunks upserts metadata containing speaker, start, and end for every chunk."""
    from pipeline.embed import embed_chunks

    chunks = [
        {"speaker": "Alice", "start": 0.0, "end": 5.0,  "text": "Hello world."},
        {"speaker": "Bob",   "start": 5.0, "end": 10.0, "text": "Hi there."},
    ]
    mock_collection = MagicMock()
    mock_model = MagicMock()
    mock_model.encode.return_value = _mock_encode([[0.0] * 1024, [0.0] * 1024])

    embed_chunks(chunks, mock_collection, mock_model, audio_filename="test.mp3")

    _, kwargs = mock_collection.upsert.call_args
    for meta in kwargs["metadatas"]:
        assert "speaker" in meta, "speaker missing from chunk metadata"
        assert "start"   in meta, "start missing from chunk metadata"
        assert "end"     in meta, "end missing from chunk metadata"


# ---------------------------------------------------------------------------
# test_query_embedding_called_once_per_query
# ---------------------------------------------------------------------------

def test_query_embedding_called_once_per_query():
    """The embedding model is called exactly once per query_rag invocation."""
    from pipeline.rag import query_rag

    mock_collection = MagicMock()
    mock_embedding_model = MagicMock()
    mock_embedding_model.encode.return_value = _mock_encode([0.0] * 1024)
    mock_collection.query.return_value = _make_mock_results(1)

    with patch("pipeline.rag.HYDE_ENABLED", False), \
         patch("pipeline.rag.RERANKER_ENABLED", False), \
         patch("pipeline.rag._build_openai_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = _make_mock_response()

        query_rag("What happened?", mock_collection, mock_embedding_model)

    assert mock_embedding_model.encode.call_count == 1


# ---------------------------------------------------------------------------
# test_config_all_env_vars_have_defaults
# ---------------------------------------------------------------------------

def test_config_all_env_vars_have_defaults():
    """Every config value resolves to a non-None value without any environment variables set."""
    import sys
    from unittest.mock import patch

    # Suppress load_dotenv so the .env file cannot supply values
    with patch("dotenv.load_dotenv"), patch.dict("os.environ", {}, clear=True):
        for mod in list(sys.modules):
            if mod == "config" or mod.startswith("config."):
                del sys.modules[mod]
        import config

        assert config.EMBEDDING_MODEL  is not None
        assert config.CHROMA_HOST      is not None
        assert config.CHROMA_PORT      is not None
        assert config.LLM_MODEL        is not None
        assert config.MAX_TOKENS       is not None
        assert config.TOP_K_RESULTS    is not None
        assert config.QUERY_CACHE_ENABLED is not None
        assert config.VIBEVOICE_VLLM_URL is not None
        assert config.CHUNK_MAX_CHARS  is not None
        assert config.RERANKER_MODEL   is not None
        assert config.RERANKER_ENABLED is not None
        assert config.HYDE_ENABLED     is not None
        assert config.GROQ_API_KEY      is not None  # defaults to ""

    # Restore original config so later tests are unaffected
    for mod in list(sys.modules):
        if mod == "config" or mod.startswith("config."):
            del sys.modules[mod]
