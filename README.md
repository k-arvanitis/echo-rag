<!-- TODO: update GitHub repo description manually in repo Settings > About to:
"Meeting intelligence RAG — speaker-aware retrieval with timestamps, HyDE, cross-encoder reranking, VibeVoice-ASR (local single-pass STT+diarization), ChromaDB, Groq." -->
[![CI](https://github.com/k-arvanitis/echo-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/k-arvanitis/echo-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=for-the-badge&logoColor=white)


https://github.com/user-attachments/assets/25a15728-b5fb-4471-ba13-a06e42234328

## Demo

The recording above walks through the four main tabs generated automatically on upload:

| Tab / Feature | What it demonstrates |
|---|---|
| Overview tab | Auto-generated 2-3 paragraph summary — main topics, key points, speaker attribution |
| Show Notes tab | Timestamped chapters and notable quotes per speaker |
| Transcript tab | Full speaker-labelled table, filterable and scrollable |
| Ask tab | Free-form question — answer streams in real time, Sources expands to show retrieved chunks with speaker label, timestamp range, and inline audio player seeked to that exact moment |

The inline audio player in Sources is seeked automatically to the timestamp of each retrieved chunk — clicking a source plays the exact moment in the recording that the answer is drawn from.

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

Yes — by "architecture diagram" I mean a proper figure. Add one exported PNG/SVG under `assets/` and embed it here later. Good options: Excalidraw, Figma, or diagrams.net.

Suggested figure sections:
- Upload + Streamlit session boundary
- VibeVoice transcription service
- Chunking + speaker naming
- Embedding + Chroma indexing
- Retrieval path: HyDE → ANN → reranker → Groq answer generation
- Output surfaces: summary, show notes, transcript, ask tab

Current text diagram:

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

## Constraints and Engineering Decisions

This project is intentionally framed as an engineering tradeoff exercise, not just a model demo. The interesting part was deciding what to optimize for under real constraints: long-form audio, speaker attribution, local-first processing, portfolio-grade UX, and a setup simple enough for reproducible demos.

### 1. Constraint: speaker-aware retrieval had to be trustworthy, not just semantically plausible

A generic transcript QA system can answer topical questions, but it often loses *who* said something and *when*. For meeting intelligence, that is the product.

**Options considered**
- Fixed-size chunking for simplicity
- Post-hoc attribution after retrieval
- Speaker-turn-aware chunking before indexing

**Decision**
- Index speaker-turn-aware chunks and preserve speaker/timestamp metadata end-to-end.

**Why this tradeoff was worth it**
- It keeps the semantic unit aligned with the actual conversation structure.
- It avoids answers that are topically relevant but lose attribution.
- It makes source display much stronger in demos because every answer can point to a person and exact moment.

### 2. Constraint: diarization alignment complexity would create more failure modes than value

A common stack is ASR first, diarization second, then timestamp/speaker alignment. That is flexible, but it adds extra model orchestration and extra points of failure.

**Options considered**
- Separate ASR + diarization pipeline with post-processing alignment
- Single-pass model that emits text, timestamps, and speaker IDs together

**Decision**
- Use VibeVoice-ASR via vLLM for single-pass transcription + diarization.

**Why this tradeoff was worth it**
- It removes a full alignment stage.
- It reduces pipeline complexity for a demoable system.
- It avoids transcription/diarization drift on long recordings.
- The downside is higher hardware demand, which is acceptable for this project because correctness and architecture clarity mattered more than commodity-laptop portability.

### 3. Constraint: long recordings exceed practical model context limits

Meeting and interview recordings are often much longer than what a single transcription request should safely handle.

**Options considered**
- Truncate audio and optimize for short demos only
- Push full recordings through one oversized request
- Split deterministically, preserve offsets, and reassemble cleanly

**Decision**
- Chunk long audio into fixed-duration windows, transcribe each chunk, and reapply offsets during reconstruction.

**Why this tradeoff was worth it**
- It preserves correctness on long inputs.
- It keeps failure domains smaller.
- It created a clean place to optimize later; the implementation now streams chunk creation from disk instead of loading the entire recording into memory first, and uses bounded parallelism for the long-audio path.

### 4. Constraint: vague conversational queries perform poorly with raw embedding lookup alone

Questions like "what did we commit to?" or "what objections came up?" often do not lexically resemble the best answer span.

**Options considered**
- Plain dense retrieval only
- Query expansion via HyDE
- Heavier retrieval stack with more complex orchestration

**Decision**
- Use HyDE as a targeted retrieval enhancement, then rerank with a cross-encoder.

**Why this tradeoff was worth it**
- HyDE improves recall for abstract conversational queries.
- The reranker improves precision before generation.
- The cost is extra latency, but that is explicit and configurable via `HYDE_ENABLED` and `RERANKER_ENABLED`.
- For a senior-level framing, this is an important point: retrieval quality was treated as a measurable systems problem, not delegated blindly to the LLM.

### 5. Constraint: portfolio UX needed to feel product-like, not notebook-like

A strong portfolio project has to demonstrate not just ML components, but the experience of using them.

**Options considered**
- CLI-only demo
- Thin app layer with upload, review, transcript, and QA surfaces
- A more custom frontend with higher implementation cost

**Decision**
- Use Streamlit to ship a usable end-to-end interface quickly.

**Why this tradeoff was worth it**
- It made it possible to demonstrate the full workflow: upload, transcription, speaker naming, summary, show notes, transcript browsing, and grounded QA.
- It is not the final frontend architecture I would choose for production, but it was the right tool for maximizing product surface area per unit of implementation time.

### 6. Constraint: multi-user isolation was needed without building a full auth/data model

Even in demo mode, different users should not collide in the same vector index.

**Options considered**
- One shared collection for all sessions
- Full user/account model
- Lightweight per-session collection isolation

**Decision**
- Scope Chroma collections by session ID.

**Why this tradeoff was worth it**
- It prevents index interference with very little infrastructure.
- It keeps the local demo simple.
- The tradeoff is that session identity is ephemeral and not production-grade, which is acceptable because the architecture leaves room to swap this for durable user/job state later.

### 7. Constraint: demo reproducibility mattered more than production-scale infrastructure

The system needed to be easy to clone, run, and explain.

**Options considered**
- Managed vector DB and hosted transcription/generation stack
- Fully local stack with minimal moving parts
- Microservice-heavy deployment from day one

**Decision**
- Keep Chroma local, keep transcription local, and use Groq only for generation-oriented tasks.

**Why this tradeoff was worth it**
- It keeps the setup understandable and reproducible.
- It makes the privacy boundary explicit: raw audio stays local, while transcript text is sent to Groq for summaries and answers.
- It also makes the migration path clear: if stricter privacy is required, the generation layer can be swapped for a local endpoint.

### What makes this senior-level work

The value in this project is not that it uses many components. It is that each component exists because of a specific constraint, and each choice carries an explicit tradeoff.

The system was designed around:
- preserving attribution, not just topical relevance
- reducing pipeline failure modes, not just maximizing modularity
- handling long inputs safely, not just optimizing for the happy path
- making retrieval quality a first-class concern
- keeping the demo architecture small without pretending it is production-complete

That framing is the difference between "I built an AI app" and "I made deliberate architecture decisions under product and systems constraints."
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

## Benchmarks

Fill these in after you run your own measurements.

| Scenario | Input size | Hardware | Latency | Notes |
|---|---|---|---|---|
| Full ingest | TODO | TODO | TODO | transcription + naming + chunking + indexing |
| First query | TODO | TODO | TODO | includes HyDE + reranking |
| Cached repeat query | TODO | TODO | TODO | same session, same prompt |
| Long-audio ingest | TODO | TODO | TODO | multi-chunk transcription path |

## Tests

```bash
uv run pytest tests/
```

All external services are fully mocked — no GPU, no running ChromaDB, no Groq API key required.

| File | What it covers |
|---|---|
| `test_chunk.py` | Character-budget chunking, speaker-turn boundary preservation, edge cases (single turn, empty input) |
| `test_rag.py` | HyDE toggle, retrieval formatting, speaker-labeled context assembly, Groq response handling |
| `test_pipeline.py` | End-to-end integration: transcribe → chunk → embed → query, fully mocked |

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

## Tradeoffs

- **Streamlit over custom frontend** — faster demo velocity and lower integration cost, but less control over complex UX.
- **ChromaDB over managed vector DB** — simple local setup and zero cloud dependency, but limited production-grade scaling and ops features.
- **Groq for generation** — fast remote inference and simple API, but transcript text leaves the local machine unless swapped for local serving.
- **HyDE + reranker enabled by default** — better recall and answer quality, but extra latency and cost per query.
- **VibeVoice local transcription** — strong privacy and single-pass diarization, but high GPU requirements reduce portability.

## Deployment notes

- **Local demo mode** — run Streamlit + Docker services on one GPU workstation. This is the intended portfolio/demo setup.
- **Semi-production path** — move Streamlit behind an auth layer, persist session/user metadata outside process memory, and run Chroma/VibeVoice as separate services.
- **Production path** — split ingestion and query into separate services, add a job queue for long transcription runs, add durable object storage for uploads, and replace per-session in-memory state with persistent user/job state.
- **Privacy-sensitive deployment** — replace Groq with a local LLM endpoint so transcript text never leaves your infrastructure.

## Known Limitations

- **GPU hard requirement** — VibeVoice-ASR-7B requires ≥24 GB VRAM. The system cannot run on CPU or consumer GPUs. To demo without a GPU, replace the STT step with a hosted API (e.g. Groq Whisper) and remove the `Dockerfile.vibevoice` dependency.
- **HyDE adds one LLM call per query** — at Groq's free-tier rate limits this occasionally adds latency under load. Disable with `HYDE_ENABLED=false` to trade recall quality for speed.
- **Cross-encoder runs on CPU** — `ms-marco-MiniLM-L-6-v2` adds ~100–200ms per reranking pass. On large transcript libraries (50+ recordings) this becomes noticeable. A GPU-accelerated reranker or a lighter model (`ms-marco-TinyBERT-L-2-v2`) would reduce this.
- **Session reference is in-memory** — ChromaDB collection names are scoped per browser session. Closing the tab loses the session reference, though indexed data persists in the Docker volume and can be re-associated manually.
- **No multi-speaker audio in overlapping speech** — VibeVoice-ASR assigns one speaker label per segment. Crosstalk or simultaneous speech is attributed to one speaker only.
