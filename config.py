"""Central configuration — all environment variable reads live here.

Every other module imports from this file instead of calling os.getenv directly.
Set values in .env (see .env.example).
"""
import os

from dotenv import load_dotenv

load_dotenv()

# --- HuggingFace (optional — used if the model repo is private or rate-limited) ---
HF_TOKEN: str = os.getenv("HF_TOKEN", "")

# --- STT (VibeVoice-ASR: single-pass transcription + diarization + timestamps) ---
STT_MODEL: str = os.getenv("STT_MODEL", "microsoft/VibeVoice-ASR")
# max tokens to generate; ~8192 covers ~30 min of speech. Increase for longer audio.
STT_MAX_NEW_TOKENS: int = int(os.getenv("STT_MAX_NEW_TOKENS", "8192"))

# --- Embeddings ---
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# --- Vector DB (ChromaDB runs as a Docker service — see docker-compose.yml) ---
CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8001"))

# --- vLLM ---
VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL: str = os.getenv("VLLM_MODEL", "Qwen/Qwen3-8B")
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "512"))

# --- RAG ---
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

# --- Chunking ---
# Number of consecutive speaker turns per chunk. 3 captures a Q+A+follow-up.
CHUNK_WINDOW_TURNS: int = int(os.getenv("CHUNK_WINDOW_TURNS", "3"))
# How many turns to advance between chunks. 1 = maximum overlap (no exchange lost).
CHUNK_STRIDE_TURNS: int = int(os.getenv("CHUNK_STRIDE_TURNS", "1"))
