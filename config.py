"""Central configuration — all environment variable reads live here.

Every other module imports from this file instead of calling os.getenv directly.
Set values in .env (see .env.example).
"""
import os

from dotenv import load_dotenv

load_dotenv()

# --- HuggingFace (optional — used if the model repo is private or rate-limited) ---
HF_TOKEN: str = os.getenv("HF_TOKEN", "")

# --- Embeddings ---
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# --- Vector DB (ChromaDB runs as a Docker service — see docker-compose.yml) ---
CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8010"))

# --- LLM (Groq API — OpenAI-compatible) ---
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "512"))

# --- RAG ---
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

# --- VibeVoice vLLM backend ---
VIBEVOICE_VLLM_URL: str = os.getenv("VIBEVOICE_VLLM_URL", "http://localhost:8001/v1")

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
