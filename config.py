"""Central configuration — all environment variable reads live here.

Every other module imports from this file instead of calling os.getenv directly.
Set values in .env (see .env.example).
"""
import os

from dotenv import load_dotenv

load_dotenv()

# --- HuggingFace (optional — used if the model repo is private or rate-limited) ---
HF_TOKEN: str = os.getenv("HF_TOKEN", "")

# --- STT backend: "vibevoice_vllm" (default), "parakeet", or "vibevoice" ---
STT_BACKEND: str = os.getenv("STT_BACKEND", "vibevoice_vllm")

# --- STT (VibeVoice-ASR: single-pass transcription + diarization + timestamps) ---
STT_MODEL: str = os.getenv("STT_MODEL", "microsoft/VibeVoice-ASR")
# Tokenizer backbone used by VibeVoice-ASR processor when tokenizer files are
# not present in the STT model repo.
STT_LANGUAGE_MODEL: str = os.getenv("STT_LANGUAGE_MODEL", "Qwen/Qwen2.5-7B")
# max tokens to generate; ~8192 covers ~30 min of speech. Increase for longer audio.
STT_MAX_NEW_TOKENS: int = int(os.getenv("STT_MAX_NEW_TOKENS", "8192"))

# --- Embeddings ---
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# --- Vector DB (ChromaDB runs as a Docker service — see docker-compose.yml) ---
CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8010"))

# --- LLM (OpenAI API) ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "512"))

# --- RAG ---
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

# --- VibeVoice vLLM backend ---
VIBEVOICE_VLLM_URL: str = os.getenv("VIBEVOICE_VLLM_URL", "http://localhost:8001/v1")

# --- Parakeet backend model IDs ---
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "openai/whisper-large-v3-turbo")
# pyannote/speaker-diarization-3.1 requires accepting the licence at huggingface.co/pyannote
# AND huggingface.co/pyannote/segmentation-3.0 before first use.
DIARIZATION_MODEL: str = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")

# --- Diarization (parakeet backend) ---
# Set expected speaker count to prevent pyannote from drifting to new labels mid-episode.
# Leave as 0 to let pyannote decide automatically.
DIARIZATION_MIN_SPEAKERS: int = int(os.getenv("DIARIZATION_MIN_SPEAKERS", "0"))
DIARIZATION_MAX_SPEAKERS: int = int(os.getenv("DIARIZATION_MAX_SPEAKERS", "0"))

# --- Chunking ---
# Max characters per chunk. Turns are never split; a single oversized turn becomes its own chunk.
# ~1500 chars ≈ 2-3 minutes of typical conversational speech.
CHUNK_MAX_CHARS: int = int(os.getenv("CHUNK_MAX_CHARS", "1500"))

# --- Retrieval enhancements ---
# Cross-encoder reranker: after fetching candidates from ChromaDB, score each
# (query, chunk) pair and return only the top-k by reranker score.
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_ENABLED: bool = os.getenv("RERANKER_ENABLED", "true").lower() == "true"

# HyDE (Hypothetical Document Embeddings): ask the LLM to write a short
# hypothetical transcript excerpt that would answer the query, then embed
# that excerpt instead of the raw query. Improves recall for vague questions.
HYDE_ENABLED: bool = os.getenv("HYDE_ENABLED", "true").lower() == "true"
