"""Character-budget chunking over complete speaker turns.

Turns are accumulated until the chunk reaches CHUNK_MAX_CHARS characters.
A turn is never split — if a single turn exceeds the budget it becomes its own chunk.
"""
from config import CHUNK_MAX_CHARS


def _format_turns(turns: list[dict]) -> str:
    return "\n".join(f"[{t['speaker']}]: {t['text']}" for t in turns)


def chunk_transcript(
    segments: list[dict],
    max_chars: int = CHUNK_MAX_CHARS,
) -> list[dict]:
    """Group speaker turns into chunks up to `max_chars` characters.

    Returns list of {text, speaker, start, end}.
    `speaker` lists all speakers present in the chunk (comma-separated).
    """
    chunks = []
    current: list[dict] = []
    current_chars = 0

    for turn in segments:
        turn_len = len(turn["text"])
        if current and current_chars + turn_len > max_chars:
            speakers = list(dict.fromkeys(t["speaker"] for t in current))
            chunks.append(
                {
                    "text": _format_turns(current),
                    "speaker": ", ".join(speakers),
                    "start": current[0]["start"],
                    "end": current[-1]["end"],
                }
            )
            current = []
            current_chars = 0
        current.append(turn)
        current_chars += turn_len

    if current:
        speakers = list(dict.fromkeys(t["speaker"] for t in current))
        chunks.append(
            {
                "text": _format_turns(current),
                "speaker": ", ".join(speakers),
                "start": current[0]["start"],
                "end": current[-1]["end"],
            }
        )

    return chunks
