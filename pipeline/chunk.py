"""Sliding-window chunking over complete speaker turns.

Chunk boundaries always fall at speaker-change points — a turn is never split.
Each chunk covers CHUNK_WINDOW_TURNS consecutive turns, advancing CHUNK_STRIDE_TURNS
at a time so adjacent chunks overlap and no exchange falls in a gap.
"""
from config import CHUNK_STRIDE_TURNS, CHUNK_WINDOW_TURNS


def _format_turns(turns: list[dict]) -> str:
    """Format a window of turns as labelled dialogue."""
    return "\n".join(f"[{t['speaker']}]: {t['text']}" for t in turns)


def chunk_transcript(
    segments: list[dict],
    window: int = CHUNK_WINDOW_TURNS,
    stride: int = CHUNK_STRIDE_TURNS,
) -> list[dict]:
    """Slide a window of `window` turns over the transcript, stepping by `stride`.

    Returns list of {text, speaker, start, end}.
    `speaker` lists all speakers present in the window (comma-separated).
    """
    chunks = []
    for i in range(0, len(segments), stride):
        turns = segments[i : i + window]
        if not turns:
            break
        speakers = list(dict.fromkeys(t["speaker"] for t in turns))
        chunks.append(
            {
                "text": _format_turns(turns),
                "speaker": ", ".join(speakers),
                "start": turns[0]["start"],
                "end": turns[-1]["end"],
            }
        )
    return chunks
