# Echo

Speaker-aware RAG over audio: upload recordings, get a searchable transcript with speaker labels, and ask questions answered from the actual spoken content. Multiple audio files accumulate in the vector DB and are queryable together.

## What it does

1. **Transcribe** — VibeVoice-ASR processes up to 60 min of audio in a single pass, returning speaker labels and timestamps with no separate diarization model.
2. **Chunk** — a sliding window over complete speaker turns captures conversational context (Q + A + follow-up). Turn boundaries are never cut.
3. **Embed** — each chunk is embedded with `bge-m3` and upserted into ChromaDB (Docker), preserving speaker, timestamps, and source file as metadata.
4. **Query** — a question is embedded, top-k chunks are retrieved, and a speaker-labeled context prompt is sent to a local vLLM server.
5. **UI** — Streamlit shows the transcript, a query box, the answer, and source chunks with speaker metadata. Previous uploads persist across sessions.

## Architecture

```
Audio file
    │
    ▼
┌─────────────────────────────────────────────────┐
│  pipeline/transcribe.py                         │
│                                                 │
│  VibeVoice-ASR (single pass)                    │
│  → speaker_id, start_time, end_time, text       │
└──────────────────────────┬──────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────┐
│  pipeline/chunk.py                              │
│  sliding window over N speaker turns            │
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
│  embed query ──▶ Chroma top-k retrieval         │
│  format context with speaker labels             │
│  POST /v1/chat/completions ──▶ vLLM (Qwen3-8B) │
└──────────────────────────┬──────────────────────┘
                           │
                           ▼
                  Answer + source chunks
                  (speakers, timestamps)
```

## Setup

### 1. Install prerequisites

| Tool | Purpose | Install |
|---|---|---|
| [uv](https://astral.sh/uv) | Python package manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [Docker](https://docs.docker.com/get-docker/) | Runs ChromaDB | See docker.com |

### 2. Clone and install Python dependencies

```bash
git clone <repo-url> audio-rag
cd audio-rag
uv sync
```

`uv sync` creates `.venv` and installs everything from `pyproject.toml`, including PyTorch from the CUDA 12.1 index. To target a different CUDA version, edit `[tool.uv.index]` in `pyproject.toml` (e.g. `cu118` or `cpu`).

### 3. Start ChromaDB

```bash
docker compose up -d chroma
```

Data persists in a named Docker volume (`chroma_data`) across restarts. To wipe all indexed audio: `docker compose down -v`.

### 4. Configure environment

```bash
cp .env.example .env
# .env is pre-filled with correct defaults — no edits needed unless you change ports or models
```

See the [Configuration](#configuration) section below for all variables.

### 5. Start the vLLM server

```bash
uv run --with vllm vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 8000
```

Must be running before starting the app or eval. To use a different model update `VLLM_MODEL` in `.env` and pass the same name to `vllm serve`.

### 6. Run the app

```bash
uv run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Running the CLI

```bash
# Transcribe and index (accumulates into ChromaDB)
uv run audio-rag path/to/recording.wav

# Transcribe, index, and answer a question
uv run audio-rag path/to/recording.wav --query "What did speaker A say about X?"

# Wipe DB before indexing (start fresh)
uv run audio-rag path/to/recording.wav --clear

# Output everything as JSON
uv run audio-rag path/to/recording.wav --query "..." --json
```

## Running the evaluation

Provide a JSON dataset:

```json
[
  {
    "question": "What did the first speaker say about the project timeline?",
    "ground_truth": "The first speaker said the project is on track for Q3."
  }
]
```

```bash
python eval/evaluate.py path/to/your_dataset.json
```

Outputs per-sample and aggregate `faithfulness` and `answer_relevancy` scores using the vLLM server as the RAGAS judge.

## Configuration

All variables live in `.env`. Defaults work out of the box if you follow the setup steps above.

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | *(empty)* | HuggingFace token — optional, only needed if you hit rate limits |
| `STT_MODEL` | `microsoft/VibeVoice-ASR` | VibeVoice-ASR model ID |
| `STT_MAX_NEW_TOKENS` | `8192` | Token budget for transcription (~30 min). Increase for longer audio |
| `CHROMA_HOST` | `localhost` | ChromaDB server host |
| `CHROMA_PORT` | `8001` | ChromaDB server port (mapped in docker-compose.yml) |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Sentence-transformers model for chunk and query embedding |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM OpenAI-compatible endpoint |
| `VLLM_MODEL` | `Qwen/Qwen3-8B` | Model name passed to vLLM |
| `TOP_K_RESULTS` | `5` | Number of chunks retrieved per query |
| `MAX_TOKENS` | `512` | Max tokens in LLM response |
| `CHUNK_WINDOW_TURNS` | `3` | Speaker turns per chunk (3 = Q + A + follow-up) |
| `CHUNK_STRIDE_TURNS` | `1` | Turns to advance between chunks (1 = max overlap) |

## Project structure

```
echo-rag/
├── app.py                  # Streamlit UI
├── run_pipeline.py         # CLI entry point
├── config.py               # All env var reads — single source of truth
├── pipeline/
│   ├── transcribe.py       # VibeVoice-ASR single-pass STT + diarization
│   ├── chunk.py            # sliding-window chunking over speaker turns
│   ├── embed.py            # bge-m3 embeddings → ChromaDB (HTTP)
│   └── rag.py              # retrieval + vLLM answer generation
├── eval/
│   └── evaluate.py         # RAGAS faithfulness + answer_relevancy CLI
├── docker-compose.yml      # ChromaDB service with persistent named volume
├── pyproject.toml          # dependencies + uv config
├── .env.example            # committed — template, safe to share
├── .env                    # not committed — your local values
├── .gitignore
└── README.md
```
