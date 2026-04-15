from pipeline.chunk import chunk_transcript


def test_chunking_respects_turn_boundaries():
    segments = [
        {"speaker": "A", "start": 0.0, "end": 1.0, "text": "hello"},
        {"speaker": "B", "start": 1.0, "end": 2.0, "text": "world"},
        {"speaker": "A", "start": 2.0, "end": 3.0, "text": "again"},
    ]

    chunks = chunk_transcript(segments, max_chars=11)

    assert len(chunks) == 2
    assert chunks[0]["text"] == "[A]: hello\n[B]: world"
    assert chunks[0]["speaker"] == "A, B"
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["end"] == 2.0
    assert chunks[1]["text"] == "[A]: again"
    assert chunks[1]["speaker"] == "A"


def test_single_long_turn_becomes_own_chunk():
    segments = [
        {"speaker": "A", "start": 0.0, "end": 5.0, "text": "x" * 20},
        {"speaker": "B", "start": 5.0, "end": 6.0, "text": "hi"},
    ]

    chunks = chunk_transcript(segments, max_chars=10)

    assert len(chunks) == 2
    assert chunks[0]["speaker"] == "A"
    assert chunks[1]["speaker"] == "B"
