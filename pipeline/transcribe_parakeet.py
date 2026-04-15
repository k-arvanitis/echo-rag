"""ASR + diarization via Whisper Large V3 Turbo + pyannote 3.1.

Parakeet handles transcription with word-level timestamps.
pyannote handles speaker diarization independently.
The two outputs are merged: each word is assigned to the speaker whose
diarization segment contains its midpoint.

Same input/output interface as pipeline/transcribe.py.
"""
import logging
import time
from typing import Any

import numpy as np
import soundfile as sf

from config import (
    DIARIZATION_MAX_SPEAKERS,
    DIARIZATION_MIN_SPEAKERS,
    DIARIZATION_MODEL,
    HF_TOKEN,
    WHISPER_MODEL,
)

logger = logging.getLogger(__name__)

_TARGET_SR = 16_000


def load_stt_model() -> tuple[Any, Any]:
    """Load Parakeet ASR pipeline and pyannote diarization pipeline.

    Returns (asr_pipeline, diarization_pipeline).
    """
    import torch
    from pyannote.audio import Pipeline
    from transformers import pipeline as hf_pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("loading %s on %s", WHISPER_MODEL, device)

    t0 = time.time()
    asr = hf_pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL,
        device=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    logger.info("Whisper loaded in %.1fs", time.time() - t0)

    t0 = time.time()
    # pyannote 3.4 calls hf_hub_download with deprecated use_auth_token across
    # multiple modules. Patch every local reference before loading.
    import huggingface_hub as _hfh
    import pyannote.audio.core.pipeline as _pp_pipeline
    import pyannote.audio.core.model as _pp_model
    import pyannote.audio.pipelines.speaker_verification as _pp_sv

    _orig = _hfh.hf_hub_download

    def _patched(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return _orig(*args, **kwargs)

    for _mod in (_pp_pipeline, _pp_model, _pp_sv):
        if hasattr(_mod, "hf_hub_download"):
            setattr(_mod, "hf_hub_download", _patched)

    # PyTorch 2.6 changed torch.load default to weights_only=True which breaks
    # pyannote checkpoints. Patch torch.load for the duration of pyannote loading.
    _orig_torch_load = torch.load
    torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})

    if HF_TOKEN:
        _hfh.login(token=HF_TOKEN)
    try:
        diarization = Pipeline.from_pretrained(DIARIZATION_MODEL)
    finally:
        torch.load = _orig_torch_load
        for _mod in (_pp_pipeline, _pp_model, _pp_sv):
            if hasattr(_mod, "hf_hub_download"):
                setattr(_mod, "hf_hub_download", _orig)
    diarization.to(torch.device(device))
    logger.info("pyannote loaded in %.1fs", time.time() - t0)

    return asr, diarization


def _resample(audio_path: str) -> tuple[np.ndarray, int]:
    """Read audio file, convert to 16kHz mono."""
    import librosa
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != _TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=_TARGET_SR)
    return audio.astype(np.float32), _TARGET_SR


def _assign_speakers(
    words: list[dict],
    diarization: Any,
) -> list[dict]:
    """Map each word to a speaker using pyannote diarization output.

    For each word, the speaker whose segment contains the word's midpoint wins.
    Words in gaps are assigned to the nearest preceding speaker.
    """
    # Build flat list of (start, end, speaker)
    segments = [
        (seg.start, seg.end, spk)
        for seg, _, spk in diarization.itertracks(yield_label=True)
    ]

    result = []
    last_speaker = segments[0][2] if segments else "SPEAKER_00"

    for word in words:
        w_start, w_end = word["timestamp"]
        if w_start is None or w_end is None:
            result.append({**word, "speaker": last_speaker})
            continue
        midpoint = (w_start + w_end) / 2

        speaker = None
        for seg_start, seg_end, spk in segments:
            if seg_start <= midpoint <= seg_end:
                speaker = spk
                break

        if speaker is None:
            speaker = last_speaker
        else:
            last_speaker = speaker

        result.append({**word, "speaker": speaker})

    return result


def _ends_sentence(text: str) -> bool:
    return bool(text.rstrip()) and text.rstrip()[-1] in ".?!"


def _group_into_segments(words: list[dict]) -> list[dict]:
    """Group consecutive same-speaker words into transcript segments,
    then merge forward any segment whose previous segment did not end a sentence."""
    if not words:
        return []

    # First pass: group by speaker
    raw: list[dict] = []
    current_speaker = words[0]["speaker"]
    current_words = [words[0]]

    for word in words[1:]:
        if word["speaker"] == current_speaker:
            current_words.append(word)
        else:
            raw.append(_emit(current_words, current_speaker))
            current_speaker = word["speaker"]
            current_words = [word]
    if current_words:
        raw.append(_emit(current_words, current_speaker))

    _MAX_SEG_SECONDS = 60.0  # never merge beyond this regardless of punctuation

    # Second pass: if a segment boundary falls mid-sentence AND the current
    # segment is short, merge into the previous one. Cap at _MAX_SEG_SECONDS.
    merged: list[dict] = []
    for seg in raw:
        prev_duration = (merged[-1]["end"] - merged[-1]["start"]) if merged else 0
        mid_sentence = merged and not _ends_sentence(merged[-1]["text"])
        if mid_sentence and prev_duration < _MAX_SEG_SECONDS:
            prev = merged[-1]
            merged[-1] = {
                "speaker": prev["speaker"],
                "start": prev["start"],
                "end": seg["end"],
                "text": (prev["text"] + " " + seg["text"]).strip(),
            }
        else:
            merged.append(seg)

    return merged


def _emit(words: list[dict], speaker: str) -> dict:
    text = " ".join(w.get("text", w.get("word", "")).strip() for w in words).strip()
    start = words[0]["timestamp"][0] or 0.0
    end = words[-1]["timestamp"][1] or words[-1]["timestamp"][0] or 0.0
    return {
        "speaker": speaker,
        "start": round(start, 2),
        "end": round(end, 2),
        "text": text,
    }


def transcribe(
    audio_path: str,
    stt_model: tuple[Any, Any] | None = None,
) -> list[dict]:
    """Transcribe audio with Parakeet + pyannote diarization.

    Returns [{speaker, start, end, text}] — same format as transcribe.py.
    """
    import tempfile
    import os

    if stt_model is None:
        stt_model = load_stt_model()
    asr, diarization = stt_model

    audio, sr = _resample(audio_path)

    # Write resampled audio to temp WAV for pyannote (needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sr)
        resampled_path = tmp.name

    try:
        t0 = time.time()
        asr_result = asr(
            {"array": audio, "sampling_rate": sr},
            return_timestamps="word",
            chunk_length_s=60,
            stride_length_s=10,
        )
        logger.info("ASR done in %.1fs", time.time() - t0)

        t0 = time.time()
        diarization_kwargs = {}
        if DIARIZATION_MIN_SPEAKERS > 0:
            diarization_kwargs["min_speakers"] = DIARIZATION_MIN_SPEAKERS
        if DIARIZATION_MAX_SPEAKERS > 0:
            diarization_kwargs["max_speakers"] = DIARIZATION_MAX_SPEAKERS
        diarization_result = diarization(resampled_path, **diarization_kwargs)
        logger.info("diarization done in %.1fs", time.time() - t0)
    finally:
        os.unlink(resampled_path)

    words = asr_result.get("chunks", [])
    if not words:
        return []

    words_with_speakers = _assign_speakers(words, diarization_result)
    return _group_into_segments(words_with_speakers)
