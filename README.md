[![CI](https://github.com/k-arvanitis/echo-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/k-arvanitis/echo-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=for-the-badge&logoColor=white)


https://github.com/user-attachments/assets/25a15728-b5fb-4471-ba13-a06e42234328


# Echo — Meeting Intelligence

Query who said what across your recorded meetings, sales calls, and interviews. Speaker-aware retrieval with timestamps.

> "What did we commit to in last Tuesday's client call?"
> "What objections came up across our last 10 sales calls?"
> "Who mentioned the Q3 budget and what did they say?"

## What it does

1. **Transcribe** — [VibeVoice-ASR-7B](https://huggingface.co/microsoft/VibeVoice-ASR) served via vLLM performs single-pass speech-to-text + speaker diarization + timestamps with no resampling needed. Recordings longer than 45 minutes are split automatically and reassembled with correct offsets.

2. **Name speakers** — After transcription, the LLM scans the opening of the conversation to suggest real names for each speaker ID. A naming form lets you confirm, correct, or fill in names that weren't mentioned — useful for internal meetings where nobody introduces themselves. Names are applied to all chunks before indexing, so every answer and source cite the person by name.

3. **Chunk** — Speaker turns are grouped by character budget (≤1500 chars). Boundaries always fall at speaker-change points so each chunk is a coherent exchange. Fixed-size chunking was rejected because speaker turns are the natural semantic unit in a conversation — cutting across them loses the question that prompted an answer.

4. **Embed** — Each chunk is embedded with `BAAI/bge-m3` and upserted into ChromaDB (Docker), with speaker name, timestamps, and source filename stored as metadata. A summary and show notes are also stored for instant retrieval.

5. **Query** — User questions optionally go through HyDE (Hypothetical Document Embeddings): the LLM first writes a short hypothetical transcript excerpt that would answer the question, and that excerpt is embedded instead of the raw query — improving recall for vague questions. Top candidates are fetched from ChromaDB, reranked with a cross-encoder, and a speaker-labeled context prompt is sent to Llama 3.3 70B via Groq.

6. **UI** — Four tabs, all generated automatically on upload:
   - **Overview** — a concise 2-3 paragraph summary of the full recording covering main topics, key points, and who said what.
   - **Show Notes** — timestamped chapters (e.g. "14:22 — Pricing objections"), notable quotes attributed to each speaker, and synthesised key takeaways.
   - **Transcript** — full speaker-labelled transcript table with start/end timestamps, filterable and scrollable.
   - **Ask** — free-form question answering. Answers stream in real time. Expanding "Sources" shows each retrieved chunk with speaker label, timestamp range, and an inline audio player seeked to that exact moment.

   The sidebar lets you upload a new recording or clear your indexed audio entirely. Each browser session gets its own isolated ChromaDB collection, so multiple users can index different recordings simultaneously without interference.

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
│  speaker-labeled context  ──▶  Llama 3.3 70B    │
└──────────────────────────┬──────────────────────┘
                           │
                           ▼
                  Answer + source chunks
                  (speakers, timestamps, audio links)
```

### Why VibeVoice

VibeVoice-ASR handles transcription and speaker diarization in a single model pass — no separate diarization pipeline, no timestamp alignment step, no pyannote dependency. Audio never leaves your machine.

## Tech stack

| Component | Role | Why |
|---|---|---|
| [VibeVoice-ASR-7B](https://huggingface.co/microsoft/VibeVoice-ASR) (vLLM) | Transcription + speaker diarization | Single-pass ASR + diarization — no separate alignment step, no pyannote dependency, speaker labels native to the model |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | Chunk embeddings | Strong multilingual dense embeddings, runs locally, no API cost at query time — suitable for confidential meeting content |
| [ChromaDB](https://www.trychroma.com/) | Vector store | Lightweight, Docker-native, no cloud dependency — keeps all transcript data on-premise; sufficient for single-tenant meeting use cases |
| Speaker-turn chunking | Segmentation | Fixed-size chunking splits mid-sentence and mid-exchange — speaker-turn boundaries preserve Q+A context naturally |
| [llama-3.3-70b](https://console.groq.com/docs/models) (Groq) | RAG answers, summaries, show notes | Fast inference via Groq's OpenAI-compatible API; swap `LLM_MODEL` in `.env` to use any Groq model without code changes |
| [Streamlit](https://streamlit.io/) | UI | Fastest path to a working UI for a pipeline demo; no frontend build step |

## Privacy & data

Audio processing and speaker diarization run locally via VibeVoice-ASR — raw audio never leaves your machine. Transcript chunks and embeddings are stored in ChromaDB running in a local Docker container.

LLM inference (RAG answers, summaries, show notes) uses Groq's API; transcript text is transmitted to Groq for these steps. To keep all data fully on-premise, point `VIBEVOICE_VLLM_URL` and `LLM_MODEL` at a local vLLM server and remove the `GROQ_API_KEY` dependency.

## Setup

### 1. Prerequisites

| Tool | Purpose | Install |
|---|---|---|
| [uv](https://astral.sh/uv) | Python package manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [Docker](https://docs.docker.com/get-docker/) | Runs ChromaDB + VibeVoice | See docker.com |
| NVIDIA GPU (≥24 GB VRAM) | VibeVoice-ASR-7B inference | — |
| Groq API key | LLM for RAG answers | [console.groq.com](https://console.groq.com/keys) |

### 2. Clone and install

```bash
git clone https://github.com/k-arvanitis/echo-rag
cd echo-rag
uv sync
```

### 3. Configure environment

```bash
cp .env.example .env
# Set GROQ_API_KEY — all other values are pre-filled with correct defaults
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

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Any model available on Groq |
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
│   ├── chunk.py                      # Character-budget chunking over speaker turns
│   ├── embed.py                      # bge-m3 embeddings → ChromaDB
│   └── rag.py                        # HyDE, retrieval, reranking, Groq LLM
├── tests/
│   ├── test_chunk.py              # Chunking unit tests
│   ├── test_rag.py                # RAG pipeline unit tests
│   └── test_pipeline.py          # Integration tests (fully mocked)
├── Dockerfile.vibevoice           # VibeVoice vLLM container image
├── vibevoice_entrypoint.sh        # Container startup script
├── docker-compose.yml             # ChromaDB + VibeVoice services
├── pyproject.toml
├── .env.example                   # Template — safe to commit
└── .env                           # Local secrets — not committed
```
