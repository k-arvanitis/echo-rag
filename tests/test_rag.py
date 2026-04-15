"""Unit tests for RAG pipeline logic — no external services required."""
import pytest

from pipeline.rag import (
    _format_context,
    _sample_chunks,
    classify_intent,
    compute_stats,
)


# ---------------------------------------------------------------------------
# classify_intent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "summarize this podcast",
    "what is this about",
    "give me an overview",
    "tell me about this episode",
    "what's this about",
])
def test_classify_intent_summary(query):
    assert classify_intent(query) == "summary"


@pytest.mark.parametrize("query", [
    "who spoke more",
    "speaking time breakdown",
    "how long is this podcast",
    "word count per speaker",
    "what's the average pace",
])
def test_classify_intent_stats(query):
    assert classify_intent(query) == "stats"


@pytest.mark.parametrize("query", [
    "what did they say about AI regulation",
    "explain the main argument",
    "what was the guest's opinion on climate",
    "did anyone mention Python",
])
def test_classify_intent_rag(query):
    assert classify_intent(query) == "rag"


# ---------------------------------------------------------------------------
# _format_context
# ---------------------------------------------------------------------------

def test_format_context_basic():
    chunks = [
        {"speaker": "Alice", "start": 0.0, "end": 5.0, "text": "Hello world"},
        {"speaker": "Bob", "start": 5.0, "end": 10.0, "text": "Hi there"},
    ]
    result = _format_context(chunks)
    assert result == "[Alice | 0.0s–5.0s]: Hello world\n[Bob | 5.0s–10.0s]: Hi there"


def test_format_context_empty():
    assert _format_context([]) == ""


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------

def _make_segments():
    return [
        {"speaker": "Alice", "start": 0.0, "end": 10.0, "text": "one two three"},
        {"speaker": "Bob", "start": 10.0, "end": 15.0, "text": "four five"},
        {"speaker": "Alice", "start": 15.0, "end": 30.0, "text": "six seven eight nine ten"},
    ]


def test_compute_stats_duration():
    stats = compute_stats(_make_segments())
    assert stats["duration"] == 30


def test_compute_stats_total_words():
    stats = compute_stats(_make_segments())
    assert stats["total_words"] == 10  # 3 + 2 + 5


def test_compute_stats_speaker_order():
    stats = compute_stats(_make_segments())
    # Alice has more talk time (25s) than Bob (5s) — should be first
    assert stats["speakers"][0]["speaker"] == "Alice"
    assert stats["speakers"][1]["speaker"] == "Bob"


def test_compute_stats_talk_pct():
    stats = compute_stats(_make_segments())
    alice = next(s for s in stats["speakers"] if s["speaker"] == "Alice")
    assert alice["talk_pct"] == pytest.approx(83.3, abs=0.2)


def test_compute_stats_empty():
    assert compute_stats([]) == {}


# ---------------------------------------------------------------------------
# _sample_chunks
# ---------------------------------------------------------------------------

def _make_chunks(n: int, chars_each: int = 100) -> list[dict]:
    return [
        {"text": "x" * chars_each, "speaker": "A", "start": float(i), "end": float(i + 1)}
        for i in range(n)
    ]


def test_sample_chunks_under_budget_returns_all():
    chunks = _make_chunks(5, chars_each=100)  # 500 chars total, well under 18k
    assert _sample_chunks(chunks) == chunks


def test_sample_chunks_over_budget_preserves_anchors():
    # 200 chunks × 200 chars = 40k chars > 18k budget
    chunks = _make_chunks(200, chars_each=200)
    sampled = _sample_chunks(chunks)
    # First 3 and last 3 must always be present
    assert chunks[0] in sampled
    assert chunks[1] in sampled
    assert chunks[2] in sampled
    assert chunks[-1] in sampled
    assert chunks[-2] in sampled
    assert chunks[-3] in sampled


def test_sample_chunks_over_budget_stays_within_budget():
    chunks = _make_chunks(200, chars_each=200)
    sampled = _sample_chunks(chunks)
    total_chars = sum(len(c["text"]) for c in sampled)
    assert total_chars <= 18_000


def test_sample_chunks_preserves_chronological_order():
    chunks = _make_chunks(200, chars_each=200)
    sampled = _sample_chunks(chunks)
    starts = [c["start"] for c in sampled]
    assert starts == sorted(starts)


def test_sample_chunks_empty():
    assert _sample_chunks([]) == []
