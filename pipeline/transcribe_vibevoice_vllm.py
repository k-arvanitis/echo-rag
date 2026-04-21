"""VibeVoice-ASR via vLLM OpenAI-compatible API.

Sends audio as base64 data URL to a running VibeVoice vLLM server
(see Dockerfile.vibevoice + docker-compose vibevoice service).

Same input/output interface as pipeline/transcribe.py:
  load_stt_model() → None  (no local model; server handles everything)
  transcribe(audio_path, stt_model) → [{speaker, start, end, text}]
"""
import base64
import json
import logging
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)

import soundfile as sf

from config import VIBEVOICE_VLLM_URL

_NON_SPEECH = re.compile(
    r"^\[(?:music|silence|noise|applause|laughter|inaudible|crosstalk|sound|"
    r"background music|background noise)\]$",
    re.IGNORECASE,
)

# VibeVoice tokenizes at ~12.5 tokens/sec; 65536-token context ≈ 87 min.
# Use 45-min chunks to stay safely within context.
_CHUNK_SECONDS = 45 * 60
_MAX_TRANSCRIBE_WORKERS = 2

_MIME_MAP = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
}


def load_stt_model() -> None:
    """No local model — VibeVoice runs as a separate vLLM service."""
    return None


def _get_duration(audio_path: str) -> float:
    """Get audio duration in seconds via ffprobe."""
    import subprocess
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
    return float(out)


def _audio_to_b64(audio_path: str) -> tuple[str, str]:
    """Read audio file, return (base64_string, mime_type)."""
    ext = os.path.splitext(audio_path)[1].lower()
    mime = _MIME_MAP.get(ext, "audio/wav")
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


def _call_api(audio_path: str, duration: float) -> list[dict]:
    """POST audio to VibeVoice vLLM and return raw parsed output."""
    from openai import APIConnectionError, APIStatusError, OpenAI

    audio_b64, mime = _audio_to_b64(audio_path)
    data_url = f"data:{mime};base64,{audio_b64}"

    show_keys = ["Start time", "End time", "Speaker ID", "Content"]
    prompt = (
        f"This is a {duration:.2f} seconds audio, please transcribe it with these keys: "
        + ", ".join(show_keys)
    )

    client = OpenAI(base_url=VIBEVOICE_VLLM_URL, api_key="dummy")
    try:
        stream = client.chat.completions.create(
            model="vibevoice",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that transcribes audio input into text output in JSON format.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            max_tokens=None,
            temperature=0.0,
            top_p=1.0,
            stream=True,
        )
    except APIConnectionError as e:
        raise RuntimeError(
            f"Cannot reach VibeVoice at {VIBEVOICE_VLLM_URL}. "
            "Is the container running? Try: docker compose up -d"
        ) from e
    except APIStatusError as e:
        raise RuntimeError(f"VibeVoice API error: {e.status_code} — {e.message}") from e

    full_text = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            full_text += delta

    return _parse_json(full_text)


def _parse_json(text: str) -> list[dict]:
    """Parse VibeVoice JSON output into a list of segment dicts."""
    # Strip markdown code fences
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        text = text[start:end if end > start else len(text)].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.rfind("```")
        text = text[start:end if end > start else len(text)].strip()

    # Locate the JSON array/object
    for opener in ("[", "{"):
        idx = text.find(opener)
        if idx != -1:
            text = text[idx:]
            break
    else:
        return []

    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        # Find balanced JSON by counting brackets
        closer = "]" if text[0] == "[" else "}"
        depth, end_idx = 0, 0
        for i, ch in enumerate(text):
            if ch == text[0]:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break
        try:
            raw = json.loads(text[:end_idx])
        except json.JSONDecodeError:
            return []

    return raw if isinstance(raw, list) else [raw]


def _parse_segments(raw: list[dict], time_offset: float = 0.0) -> list[dict]:
    """Normalise raw VibeVoice API output to [{speaker, start, end, text}]."""
    segments = []
    for seg in raw:
        text = str(seg.get("Content", seg.get("content", seg.get("text", "")))).strip()
        if not text or _NON_SPEECH.match(text):
            continue
        start = float(seg.get("Start time", seg.get("Start", seg.get("start_time", 0.0))))
        end = float(seg.get("End time", seg.get("End", seg.get("end_time", 0.0))))
        speaker = str(seg.get("Speaker ID", seg.get("Speaker", seg.get("speaker_id", "0"))))
        segments.append({
            "speaker": speaker,
            "start": round(start + time_offset, 2),
            "end": round(end + time_offset, 2),
            "text": text,
        })
    return segments


def _transcribe_chunk(chunk_path: str, chunk_duration: float, time_offset: float, index: int, total: int) -> list[dict]:
    logger.info("chunk %d/%d at %.1f min", index, total, time_offset / 60)
    try:
        t0 = time.time()
        raw = _call_api(chunk_path, chunk_duration)
        logger.info("chunk %d/%d done in %.1fs", index, total, time.time() - t0)
        return _parse_segments(raw, time_offset=time_offset)
    finally:
        os.unlink(chunk_path)



def _iter_audio_chunks(audio_path: str, chunk_seconds: int) -> tuple[int, list[tuple[str, float, float]]]:
    with sf.SoundFile(audio_path) as audio_file:
        sr = audio_file.samplerate
        channels = audio_file.channels
        chunk_frames = int(chunk_seconds * sr)
        total_frames = len(audio_file)
        chunk_specs: list[tuple[str, float, float]] = []

        for offset_frames in range(0, total_frames, chunk_frames):
            frames_to_read = min(chunk_frames, total_frames - offset_frames)
            chunk = audio_file.read(frames=frames_to_read, dtype="float32", always_2d=channels > 1)
            if channels > 1:
                chunk = chunk.mean(axis=1)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, chunk, sr)
                chunk_specs.append((tmp.name, frames_to_read / sr, offset_frames / sr))

    return len(chunk_specs), chunk_specs



def transcribe(
    audio_path: str,
    stt_model: Any = None,
) -> list[dict]:
    """Transcribe audio via VibeVoice vLLM API."""
    try:
        duration = _get_duration(audio_path)
    except Exception:
        with sf.SoundFile(audio_path) as audio_file:
            duration = len(audio_file) / audio_file.samplerate

    if duration <= _CHUNK_SECONDS:
        t0 = time.time()
        raw = _call_api(audio_path, duration)
        logger.info("VibeVoice vLLM done in %.1fs", time.time() - t0)
        return _parse_segments(raw)

    total_chunks, chunk_specs = _iter_audio_chunks(audio_path, _CHUNK_SECONDS)
    logger.info("audio is %.1f min — splitting into %d chunks", duration / 60, total_chunks)

    with ThreadPoolExecutor(max_workers=min(_MAX_TRANSCRIBE_WORKERS, total_chunks)) as executor:
        futures = [
            executor.submit(_transcribe_chunk, chunk_path, chunk_duration, time_offset, i + 1, total_chunks)
            for i, (chunk_path, chunk_duration, time_offset) in enumerate(chunk_specs)
        ]

    all_segments: list[dict] = []
    for future in futures:
        all_segments.extend(future.result())
    return all_segments
