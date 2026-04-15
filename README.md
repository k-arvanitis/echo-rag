# Echo — Meeting Intelligence RAG

Query who said what across your recorded meetings, sales calls, and interviews. Speaker-aware retrieval with timestamps.

> "What did we commit to in last Tuesday's client call?"
> "What objections came up across our last 10 sales calls?"
> "Who mentioned the Q3 budget and what did they say?"

## What it does

1. **Transcribe** — [VibeVoice-ASR-7B](https://huggingface.co/microsoft/VibeVoice-ASR) served via vLLM. Single-pass STT + speaker diarization + timestamps, no resampling needed. Handles up to 45 min per chunk; longer recordings are split automatically.
2. **Chunk** — Speaker turns are grouped by character budget (≤1500 chars). Boundaries always fall at speaker-change points so each chunk is a coherent exchange.
3. **Embed** — Each chunk is embedded with `BAAI/bge-m3` and upserted into ChromaDB (Docker), with speaker, timestamps, and source file stored as metadata.
4. **Query** — User questions optionally go through HyDE (Hypothetical Document Embeddings) before retrieval. Top candidates are fetched from ChromaDB, reranked with a cross-encoder, and a speaker-labeled context prompt is sent to GPT-4o-mini.
5. **UI** — Streamlit shows an overview summary, auto-generated show notes (chapters + quotes + takeaways), full transcript table, and a free-form Ask tab. Source chunks link back to the exact audio timestamp.

## Use cases

**Sales teams** — Index discovery calls, demos, and follow-ups. Ask "What pricing objections came up?" or "Which prospect mentioned a competitor?" across your entire call library.

**Research interviews** — Index user interviews or expert calls. Retrieve exact quotes by topic — "What did participants say about onboarding friction?" — without re-reading transcripts.

**Internal meetings** — Capture decisions and action items. Ask "What did we agree on for the Q4 roadmap?" or "Who owns the infra migration?" and get the answer with a timestamp and speaker label.

## Architecture

```
Audio file
    │
    ▼
┌─────────────────────────────────────────────────┐
│  pipeline/transcribe_vibevoice_vllm.py          │
│  VibeVoice-ASR-7B via vLLM (Docker)            │
│  single-pass: ASR + speaker IDs + timestamps   │
│  ──▶  [{speaker, start, end, text}]            │
└──────────────────────────┬──────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────┐
│  pipeline/chunk.py                              │
│  character-budget grouping of speaker turns     │
└──────────────────────────┬──────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────┐
│  pipeline/embed.py                              │
│  BAAI/bge-m3 embeddings  ──▶  ChromaDB (Docker)│
│  metadata: speaker, start, end, audio_file      │
└──────────────────────────┬──────────────────────┘
                           │
          ┌────────────────┘
          │   Query time
          ▼
┌─────────────────────────────────────────────────┐
│  pipeline/rag.py                                │
│  HyDE: hypothetical excerpt embedding           │
│  ChromaDB top-k  ──▶  cross-encoder rerank      │
│  speaker-labeled context  ──▶  GPT-4o-mini      │
└──────────────────────────┬──────────────────────┘
                           │
                           ▼
                  Answer + source chunks
                  (speakers, timestamps, audio links)
```

## Setup

### 1. Prerequisites

| Tool | Purpose | Install |
|---|---|---|
| [uv](https://astral.sh/uv) | Python package manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [Docker](https://docs.docker.com/get-docker/) | Runs ChromaDB + VibeVoice | See docker.com |
| NVIDIA GPU (≥24 GB VRAM) | VibeVoice-ASR-7B inference | — |
| OpenAI API key | GPT-4o-mini for RAG answers | [platform.openai.com](https://platform.openai.com/api-keys) |

### 2. Clone and install

```bash
git clone <repo-url> echo-rag
cd echo-rag
uv sync
```

### 3. Configure environment

```bash
cp .env.example .env
# Set OPENAI_API_KEY — all other values are pre-filled with correct defaults
```

### 4. Start services

```bash
# Download VibeVoice-ASR weights first (one-time, ~15 GB):
huggingface-cli download microsoft/VibeVoice-ASR

# Build and start ChromaDB + VibeVoice vLLM:
docker compose up -d
```

ChromaDB runs on port 8010. VibeVoice loads on port 8001 (takes ~2 min on first start).

### 5. Run

```bash
uv run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). Upload an audio file — transcription + indexing runs automatically.

## Evaluation

QA pairs are synthesized from indexed chunks using GPT-4o-mini, then judged on four [DeepEval](https://docs.confident-ai.com/) metrics:

```bash
uv run python eval/evaluate.py --user-id <session-uuid>

# Save QA pairs for re-use
uv run python eval/evaluate.py --user-id <uuid> --save eval/qa_pairs.json

# Re-use saved pairs (skip synthesis)
uv run python eval/evaluate.py --user-id <uuid> --dataset eval/qa_pairs.json
```

| Metric | What it measures |
|---|---|
| Faithfulness | Answer grounded in retrieved context? |
| Answer Relevancy | Answer addresses the question? |
| Contextual Precision | Top chunks are the most relevant? |
| Contextual Recall | Chunks contain what's needed to answer? |

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model for RAG answers and show notes |
| `STT_BACKEND` | `vibevoice_vllm` | `vibevoice_vllm` (default), `parakeet` (Whisper+pyannote), or `vibevoice` (in-process) |
| `VIBEVOICE_VLLM_URL` | `http://localhost:8001/v1` | VibeVoice vLLM endpoint |
| `VIBEVOICE_GPU_UTIL` | `0.60` | GPU memory fraction for VibeVoice container |
| `CHROMA_HOST` | `localhost` | ChromaDB host |
| `CHROMA_PORT` | `8010` | ChromaDB port |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Sentence-transformers embedding model |
| `TOP_K_RESULTS` | `5` | Chunks returned per query (after reranking) |
| `MAX_TOKENS` | `512` | Max tokens in LLM response |
| `CHUNK_MAX_CHARS` | `1500` | Max characters per transcript chunk |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `RERANKER_ENABLED` | `true` | Enable reranking after retrieval |
| `HYDE_ENABLED` | `true` | Enable Hypothetical Document Embeddings |

## Project structure

```
echo-rag/
├── app.py                         # Streamlit UI
├── config.py                      # All env var reads — single source of truth
├── pipeline/
│   ├── transcribe_vibevoice_vllm.py  # VibeVoice-ASR via vLLM (default STT)
│   ├── transcribe_parakeet.py        # Whisper Large V3 Turbo + pyannote
│   ├── transcribe.py                 # VibeVoice-ASR in-process
│   ├── chunk.py                      # Character-budget chunking over speaker turns
│   ├── embed.py                      # bge-m3 embeddings → ChromaDB
│   └── rag.py                        # HyDE, retrieval, reranking, GPT-4o-mini
├── eval/
│   └── evaluate.py                # DeepEval: 4-metric RAG evaluation
├── tests/
│   └── test_rag.py                # Unit tests for RAG pipeline
├── Dockerfile.vibevoice           # VibeVoice vLLM container image
├── vibevoice_entrypoint.sh        # Container startup script
├── docker-compose.yml             # ChromaDB + VibeVoice services
├── pyproject.toml
├── .env.example                   # Template — safe to commit
└── .env                           # Local secrets — not committed
```
