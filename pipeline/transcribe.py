"""Single-pass STT + diarization with VibeVoice-ASR → speaker-labeled segments.

VibeVoice-ASR processes up to 60 min of audio in one forward pass and returns
structured segments with speaker ID and timestamps — no separate diarization
model required.
"""
from typing import Any

import soundfile as sf

from config import STT_MAX_NEW_TOKENS, STT_MODEL


def load_stt_model() -> tuple[Any, Any]:
    """Load VibeVoice-ASR processor and model. Returns (processor, model).

    Requires transformers >= 4.51 for VibeVoiceASR class support.
    trust_remote_code=True is mandatory — the model ships its own processor
    and generation logic.
    """
    from transformers import (
        VibeVoiceASRForConditionalGeneration,
        VibeVoiceASRProcessor,
    )

    processor = VibeVoiceASRProcessor.from_pretrained(
        STT_MODEL, trust_remote_code=True
    )
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        STT_MODEL,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return processor, model


def _generate(processor: Any, model: Any, audio_path: str) -> list[dict]:
    """Run VibeVoice-ASR inference on an audio file; return raw segment dicts."""
    audio, sr = sf.read(audio_path)

    inputs = processor(
        audio=[(audio, sr)],
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=STT_MAX_NEW_TOKENS,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        do_sample=False,
        num_beams=1,
    )

    # VibeVoice-ASR encodes audio as features (not input_ids), so the full
    # output tensor is the generated transcript token sequence.
    n_input = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    generated_ids = output_ids[0][n_input:]
    raw_text = processor.decode(generated_ids, skip_special_tokens=True)
    return processor.post_process_transcription(raw_text)


def transcribe(
    audio_path: str,
    stt_model: tuple[Any, Any] | None = None,
) -> list[dict]:
    """Transcribe audio in a single pass. Returns [{speaker, start, end, text}].

    Pass a pre-loaded (processor, model) tuple to avoid reloading between calls.
    """
    if stt_model is None:
        stt_model = load_stt_model()
    processor, model = stt_model

    raw_segments = _generate(processor, model, audio_path)

    return [
        {
            "speaker": str(seg["speaker_id"]),
            "start": round(float(seg["start_time"]), 2),
            "end": round(float(seg["end_time"]), 2),
            "text": seg["text"].strip(),
        }
        for seg in raw_segments
        if seg.get("text", "").strip()
    ]
