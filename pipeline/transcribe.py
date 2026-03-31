"""Single-pass STT + diarization with VibeVoice-ASR → speaker-labeled segments.

VibeVoice-ASR processes up to 60 min of audio in one forward pass and returns
structured segments with speaker ID and timestamps — no separate diarization
model required.
"""
from typing import Any

import soundfile as sf
from huggingface_hub import hf_hub_download

from config import STT_LANGUAGE_MODEL, STT_MAX_NEW_TOKENS, STT_MODEL

_FALLBACK_CHAT_TEMPLATE = """{%- set system_prompt = system_prompt | default("You are a helpful assistant that transcribes audio input into text output in JSON format.") -%}
<|im_start|>system
{{ system_prompt }}<|im_end|>
{%- set audio_token = audio_token | default("<|box_start|>") -%}
{%- set audio_start_token = "<|object_ref_start|>" -%}
{%- set audio_end_token = "<|object_ref_end|>" -%}
{%- for message in messages -%}
    {%- if message['role'] == 'user' -%}
{{ '\n' }}<|im_start|>user{{ '\n' }}{%- set text_items = message['content'] | selectattr('type', 'equalto', 'text') | list -%}
        {%- set context_text = text_items[0]['text'] if text_items else none -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'audio' -%}
{{ audio_start_token }}{{ audio_token }}{{ audio_end_token }}{{ "\n" }}{%- if context_text -%}
This is a <|AUDIO_DURATION|> seconds audio, with extra info: {{ context_text }}

Please transcribe it with these keys: Start time, End time, Speaker ID, Content{%- else -%}
This is a <|AUDIO_DURATION|> seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content{%- endif -%}
            {%- endif -%}
        {%- endfor -%}
<|im_end|>{{ '\n' }}
    {%- endif -%}
{%- endfor -%}
"""


def _ensure_chat_template(processor: Any) -> Any:
    """Attach a chat template required by apply_transcription_request()."""
    if getattr(processor, "chat_template", None):
        return processor

    for model_id in (STT_MODEL, "microsoft/VibeVoice-ASR-HF"):
        try:
            path = hf_hub_download(model_id, "chat_template.jinja")
            with open(path, encoding="utf-8") as f:
                processor.chat_template = f.read()
            return processor
        except Exception:
            continue

    processor.chat_template = _FALLBACK_CHAT_TEMPLATE
    return processor


def _resolve_transformers_model_id(model_id: str) -> str:
    """Map legacy repo IDs to a transformers-compatible ASR checkpoint."""
    if model_id == "microsoft/VibeVoice-ASR":
        return "microsoft/VibeVoice-ASR-HF"
    return model_id


def _load_vibevoice_from_package() -> tuple[Any, Any]:
    """Load processor/model from the official microsoft/VibeVoice package."""
    from vibevoice.modular.modeling_vibevoice_asr import (
        VibeVoiceASRForConditionalGeneration,
    )
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

    processor = VibeVoiceASRProcessor.from_pretrained(
        STT_MODEL,
        language_model_pretrained_name=STT_LANGUAGE_MODEL,
    )
    processor = _ensure_chat_template(processor)
    import torch
    cuda_available = torch.cuda.is_available()
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        STT_MODEL,
        dtype=torch.bfloat16 if cuda_available else torch.float32,
        trust_remote_code=True,
    )
    if cuda_available:
        model = model.cuda()
    model.eval()
    return processor, model


def _load_vibevoice_from_transformers() -> tuple[Any, Any]:
    """Load processor/model from transformers only (no vibevoice package)."""
    from transformers import (
        AutoProcessor,
        AutoTokenizer,
        VibeVoiceAcousticTokenizerFeatureExtractor,
        VibeVoiceAsrForConditionalGeneration,
        VibeVoiceAsrProcessor,
    )

    import torch
    model_id = _resolve_transformers_model_id(STT_MODEL)
    cuda_available = torch.cuda.is_available()
    print(f"[stt] cuda_available={cuda_available}")

    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        # microsoft/VibeVoice-ASR does not ship processor/tokenizer files, so we
        # construct the processor explicitly.
        feature_extractor = VibeVoiceAcousticTokenizerFeatureExtractor()
        tokenizer = AutoTokenizer.from_pretrained(STT_LANGUAGE_MODEL)
        processor = VibeVoiceAsrProcessor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )
    processor = _ensure_chat_template(processor)

    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cuda_available else torch.float32,
    )
    if cuda_available:
        model = model.cuda()
    model.eval()
    return processor, model


def load_stt_model() -> tuple[Any, Any]:
    """Load VibeVoice-ASR processor and model. Returns (processor, model)."""
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[stt] GPU free before load: {torch.cuda.mem_get_info()[0] / 1024**3:.1f} GB")

    try:
        processor, model = _load_vibevoice_from_transformers()
        print(
            f"[stt] loader=transformers model_id={_resolve_transformers_model_id(STT_MODEL)} "
            f"model_class={type(model).__name__}"
        )
        return processor, model
    except Exception as exc:
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[stt] transformers loader failed: {exc}")

    try:
        processor, model = _load_vibevoice_from_package()
        print(f"[stt] loader=vibevoice model_id={STT_MODEL} model_class={type(model).__name__}")
        return processor, model
    except Exception as exc:
        print(f"[stt] vibevoice package loader failed: {exc}")
        raise RuntimeError(
            "Failed to load VibeVoice-ASR.\n"
            "Ensure transformers has VibeVoice ASR support and set "
            "STT_MODEL/STT_LANGUAGE_MODEL correctly in .env."
        ) from exc


def _generate(processor: Any, model: Any, audio_path: str) -> list[dict]:
    """Run VibeVoice-ASR inference on an audio file; return raw segment dicts."""
    # Always derive device from actual parameters — device_map="auto" models
    # don't have a reliable .device attribute and can silently return cpu.
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    print(f"[stt] device={model_device} dtype={model_dtype} audio={audio_path}")

    def _prepare_inputs_for_model(batch: Any) -> Any:
        prepared = {}
        for key, value in batch.items():
            if hasattr(value, "to"):
                if getattr(value, "is_floating_point", lambda: False)() and model_dtype is not None:
                    prepared[key] = value.to(device=model_device, dtype=model_dtype)
                else:
                    prepared[key] = value.to(device=model_device)
            else:
                prepared[key] = value
        return prepared

    # New VibeVoice processor API expects a text prompt internally and provides
    # a helper for transcription requests.
    if hasattr(processor, "apply_transcription_request"):
        inputs = processor.apply_transcription_request(
            audio=audio_path,
        )
    else:
        audio, sr = sf.read(audio_path)
        inputs = processor(
            audio=[(audio, sr)],
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        )

    inputs = _prepare_inputs_for_model(inputs)
    print(f"[stt] inputs on: { {k: v.device if hasattr(v, 'device') else type(v).__name__ for k, v in inputs.items()} }")

    tok = processor.tokenizer
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id

    import time
    t0 = time.time()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=STT_MAX_NEW_TOKENS,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        do_sample=False,
        num_beams=1,
    )

    print(f"[stt] generate took {time.time() - t0:.1f}s")
    n_input = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    generated_ids = output_ids[:, n_input:]

    # Transformers-native VibeVoice supports direct parsed output.
    try:
        parsed = processor.decode(generated_ids, return_format="parsed")
        if isinstance(parsed, list) and parsed:
            if isinstance(parsed[0], list):
                return parsed[0]
            return parsed
    except Exception:
        pass

    # Backward-compatible path with raw text post-processing.
    raw_text = processor.decode(generated_ids[0], skip_special_tokens=True)
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
            "speaker": str(seg.get("speaker_id", seg.get("Speaker", ""))),
            "start": round(float(seg.get("start_time", seg.get("Start", 0.0))), 2),
            "end": round(float(seg.get("end_time", seg.get("End", 0.0))), 2),
            "text": str(seg.get("text", seg.get("Content", ""))).strip(),
        }
        for seg in raw_segments
        if str(seg.get("text", seg.get("Content", ""))).strip()
    ]
