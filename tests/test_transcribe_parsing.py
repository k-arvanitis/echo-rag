from pipeline.transcribe_vibevoice_vllm import _parse_json, _parse_segments
from pipeline.transcribe import _parse_segments as _parse_vibevoice_segments


def test_parse_json_strips_code_fence():
    text = """
```json
[{"Start time": 0.0, "End time": 1.0, "Speaker ID": "S0", "Content": "Hi"}]
```
"""
    parsed = _parse_json(text)
    assert parsed == [{"Start time": 0.0, "End time": 1.0, "Speaker ID": "S0", "Content": "Hi"}]


def test_parse_segments_filters_non_speech_and_offsets():
    raw = [
        {"Start time": 0.0, "End time": 1.0, "Speaker ID": "S0", "Content": "[music]"},
        {"Start time": 1.0, "End time": 2.5, "Speaker ID": "S1", "Content": "Hello"},
    ]

    segments = _parse_segments(raw, time_offset=10.0)

    assert segments == [
        {"speaker": "S1", "start": 11.0, "end": 12.5, "text": "Hello"}
    ]


def test_vibevoice_parse_segments_normalizes_keys():
    raw = [
        {"speaker_id": "A", "start_time": 0.5, "end_time": 1.0, "text": "Hi"},
        {"Speaker": "B", "Start": 1.0, "End": 2.0, "Content": "There"},
        {"speaker_id": "C", "start_time": 2.0, "end_time": 2.5, "text": "[silence]"},
    ]

    segments = _parse_vibevoice_segments(raw, time_offset=5.0)

    assert segments == [
        {"speaker": "A", "start": 5.5, "end": 6.0, "text": "Hi"},
        {"speaker": "B", "start": 6.0, "end": 7.0, "text": "There"},
    ]
