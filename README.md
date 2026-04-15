# Echo — Meeting Intelligence RAG

Query who said what across your recorded meetings, sales calls, and interviews. Speaker-aware retrieval with timestamps.

> "What did we commit to in last Tuesday's client call?"
> "What objections came up across our last 10 sales calls?"
> "Who mentioned the Q3 budget and what did they say?"

## What it does

1. **Transcribe** — Three interchangeable backends (set `STT_BACKEND` in `.env`):
   - **`parakeet`** (default) — Whisper Large V3 Turbo for ASR + pyannote 3.1 for speaker diarization. Fast, low VRAM (~2 GB), handles any length audio.
   - **`vibevoice_vllm`** — VibeVoice-ASR-7B served via vLLM (Docker). Single-pass STT + diarization, no resampling needed. Audio is sent as base64 to the vLLM endpoint. Chunks at 45 min.
   - **`vibevoice`** — VibeVoice-ASR-7B loaded in-process. Audio longer than 25 min is chunked and stitched to bypass the 60-min context limit.
2. **Chunk** — speaker turns are grouped by character budget (≤1500 chars). Turn boundaries are never cut, so each chunk is a coherent conversational exchange.
3. **Embed** — each chunk is embedded with `BAAI/bge-m3` and upserted into ChromaDB (Docker), preserving speaker, timestamps, and source file as metadata.
4. **Query** — intent is classified first (summary / speaker stats / RAG). For RAG queries, the question optionally goes through HyDE (Hypothetical Document Embeddings) before retrieval, top candidates are fetched from ChromaDB, reranked with a cross-encoder, and a speaker-labeled context prompt is streamed from a local vLLM server (Qwen3-8B). A speaker filter lets you scope retrieval to specific participants.
5. **UI** — Streamlit shows an overview, auto-generated meeting notes (chapters + quotes + takeaways), full transcript, and an Ask tab. Source chunks link to the exact audio timestamp.

## Use cases

**Sales teams**
Index discovery calls, demos, and follow-ups. Ask "What pricing objections came up?" or "Which prospect mentioned a competitor?" across your entire call library.

**Research interviews**
Index user interviews or expert calls. Retrieve exact quotes by topic — "What did participants say about onboarding friction?" — without manually re-reading transcripts.

**Internal standups and planning meetings**
Capture decisions and action items. Ask "What did we agree on for the Q4 roadmap?" or "Who owns the infra migration?" and get the answer with a timestamp and speaker label.

## Architecture

```
Audio file
    │
    ▼
┌─────────────────────────────────────────────────┐
│  pipeline/transcribe_parakeet.py  (default)     │
│  Whisper Large V3 Turbo  ──▶  ASR + timestamps  │
│  pyannote 3.1            ──▶  speaker labels    │
│  merged  ──▶  speaker_id, start, end, text      │
├─────────────────────────────────────────────────┤
│  pipeline/transcribe.py  (STT_BACKEND=vibevoice)│
│  VibeVoice-ASR-7B (single-pass, chunked >25min) │
└──────────────────────────┬──────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────┐
│  pipeline/chunk.py                              │
│  character-budget grouping of speaker turns     │
│  boundaries always at speaker-change points     │
└──────────────────────────┬──────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────┐
│  pipeline/embed.py                              │
│  bge-m3 embeddings  ──▶  ChromaDB (Docker)      │
│  metadata: speaker, start, end, audio_file      │
└──────────────────────────┬──────────────────────┘
                           │
          ┌────────────────┘
          │   Query time
          ▼
┌─────────────────────────────────────────────────┐
│  pipeline/rag.py                                │
│  intent classifier (summary / stats / RAG)      │
│  embed query ──▶ Chroma top-k retrieval         │
│  format context with speaker labels             │
│  POST /v1/chat/completions ──▶ vLLM (Qwen3-8B) │
└──────────────────────────┬──────────────────────┘
                           │
                           ▼
                  Answer + source chunks
                  (speakers, timestamps, audio links)
```

## Setup

### 1. Install prerequisites

| Tool | Purpose | Install |
|---|---|---|
| [uv](https://astral.sh/uv) | Python package manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [Docker](https://docs.docker.com/get-docker/) | Runs ChromaDB + vLLM | See docker.com |

### 2. Clone and install Python dependencies

```bash
git clone <repo-url> echo-rag
cd echo-rag
uv sync
```

### 3. Configure environment

```bash
cp .env.example .env
# Pre-filled with correct defaults — edit only if you change ports or models
```

### 4. Start services

```bash
docker compose up -d
```

This starts ChromaDB (port 8010) and vLLM serving Qwen3-8B (port 8000). Model weights (~16 GB) are downloaded on first run into `~/.cache/huggingface`.

### 5. Run the app

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Evaluation

QA pairs are synthesized from indexed chunks using GPT-4o-mini, then judged on four DeepEval metrics:

```bash
# Generate QA pairs, run eval, print results
uv run python eval/evaluate.py --user-id <session-uuid>

# Save generated QA pairs for re-use
uv run python eval/evaluate.py --user-id <uuid> --save eval/qa_pairs.json

# Re-use saved QA pairs (skip synthesis)
uv run python eval/evaluate.py --user-id <uuid> --dataset eval/qa_pairs.json
```

Requires `OPENAI_API_KEY` in `.env` for GPT-4o-mini synthesis and judging.

| Metric | What it measures |
|---|---|
| Faithfulness | Answer grounded in retrieved context? |
| Answer Relevancy | Answer addresses the question? |
| Contextual Precision | Top chunks are the most relevant? |
| Contextual Recall | Chunks contain what's needed to answer? |

The eval harness supports any JSON dataset of `{question, ground_truth}` pairs. For meeting intelligence use cases, questions like "What did speaker A commit to?" or "Was a deadline mentioned?" are good test cases for faithfulness scoring.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | *(empty)* | HuggingFace token — optional |
| `STT_BACKEND` | `parakeet` | `parakeet` (Whisper + pyannote), `vibevoice_vllm` (VibeVoice vLLM), or `vibevoice` (in-process) |
| `VIBEVOICE_VLLM_URL` | `http://localhost:8001/v1` | VibeVoice vLLM endpoint (vibevoice_vllm backend only) |
| `VIBEVOICE_GPU_UTIL` | `0.30` | GPU memory fraction for VibeVoice vLLM container (~14 GB) |
| `STT_MODEL` | `microsoft/VibeVoice-ASR` | VibeVoice-ASR model ID (vibevoice backend only) |
| `STT_MAX_NEW_TOKENS` | `8192` | Token budget per 25-min chunk (vibevoice backend only) |
| `WHISPER_MODEL` | `openai/whisper-large-v3-turbo` | ASR model (parakeet backend) |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-3.1` | Diarization model (parakeet backend) |
| `CHROMA_HOST` | `localhost` | ChromaDB server host |
| `CHROMA_PORT` | `8010` | ChromaDB server port |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM OpenAI-compatible endpoint |
| `VLLM_MODEL` | `Qwen/Qwen3-8B` | Model name as registered in vLLM |
| `TOP_K_RESULTS` | `5` | Chunks returned per query (after reranking) |
| `MAX_TOKENS` | `512` | Max tokens in LLM response |
| `CHUNK_MAX_CHARS` | `1500` | Max characters per transcript chunk |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model for reranking |
| `RERANKER_ENABLED` | `true` | Enable cross-encoder reranking after retrieval |
| `HYDE_ENABLED` | `true` | Enable Hypothetical Document Embeddings for retrieval |

## Project structure

```
echo-rag/
├── app.py                  # Streamlit UI
├── config.py               # All env var reads — single source of truth
├── pipeline/
│   ├── transcribe.py              # VibeVoice-ASR STT + diarization (in-process)
│   ├── transcribe_parakeet.py     # Whisper Large V3 Turbo + pyannote diarization
│   ├── transcribe_vibevoice_vllm.py  # VibeVoice-ASR via vLLM API (base64 audio)
│   ├── chunk.py            # character-budget chunking over speaker turns
│   ├── embed.py            # bge-m3 embeddings → ChromaDB (HTTP)
│   └── rag.py              # intent routing, retrieval, reranking, vLLM generation
├── eval/
│   └── evaluate.py         # DeepEval: GPT-4o-mini synthesis + 4-metric judging
├── Dockerfile.vibevoice    # VibeVoice vLLM container image
├── vibevoice_entrypoint.sh # Container startup: install plugin → start vLLM
├── docker-compose.yml      # ChromaDB + vLLM (Qwen3) + vibevoice services
├── pyproject.toml          # dependencies + uv config
├── .env.example            # template, safe to commit
└── .env                    # local values, not committed
```
