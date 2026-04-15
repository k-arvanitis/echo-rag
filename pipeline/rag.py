"""Retrieve relevant chunks and generate answers via vLLM."""
import json
import logging
import re
from collections.abc import Iterator

logger = logging.getLogger(__name__)

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from config import MAX_TOKENS, TOP_K_RESULTS, VLLM_BASE_URL, VLLM_MODEL

_SYSTEM_PROMPT = (
    "You are an assistant that answers questions about audio transcripts. "
    "Use only the provided transcript context. "
    "When referring to speakers, always use their name — never assume gender or use gendered pronouns."
)

_SUMMARY_SYSTEM_PROMPT = (
    "You are given the full transcript of a podcast or audio recording. "
    "Write a concise 2-3 paragraph summary covering the main topics discussed, "
    "key points made, and who said what where relevant. "
    "Be specific and informative — avoid generic statements."
)

_SHOW_NOTES_PROMPT = """You are given chunked segments of a podcast, each with a timestamp and speaker labels.
Generate show notes as JSON with this exact structure:
{
  "chapters": [{"title": "...", "start": <seconds as float>, "summary": "one sentence describing what this section is about"}],
  "quotes": [{"speaker": "...", "text": "...", "start": <seconds as float>}],
  "takeaways": ["...", "..."]
}
Rules:
- 3-6 chapters — each should represent a distinct topic shift, not just a new speaker turn
- Chapter titles must be descriptive (e.g. "The case against AI regulation") not generic (e.g. "Discussion begins")
- 3-5 quotes — pick the most insightful or memorable statements, paraphrase if needed to make them punchy
- 5-7 takeaways — synthesized insights, not copied sentences. Each should teach the reader something.
- Return only valid JSON, no markdown fences, no other text"""


def _build_openai_client() -> OpenAI:
    """Create an OpenAI client pointed at the local vLLM server."""
    return OpenAI(base_url=VLLM_BASE_URL, api_key="dummy")


# Qwen3 defaults to "thinking" mode which emits <think>…</think> before every response.
# Disable it so RAG answers and JSON outputs are clean.
_NO_THINK: dict = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

# ~4500 tokens; leaves room for system prompt + response within 8k context
_SHOW_NOTES_CHAR_BUDGET = 18_000


def retrieve(
    query: str,
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    top_k: int = TOP_K_RESULTS,
) -> list[dict]:
    """Embed query and return top-k chunks with their speaker metadata."""
    query_embedding = embedding_model.encode(query).tolist()
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k + 2,  # fetch extra to account for filtered-out meta docs
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        raise RuntimeError(f"ChromaDB query failed: {e}") from e

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if meta.get("type") in ("summary", "show_notes"):
            continue
        chunks.append({**meta, "text": doc, "distance": round(dist, 4)})
    return chunks[:top_k]


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a labelled context block for the prompt."""
    lines = [
        f"[{c['speaker']} | {c['start']:.1f}s–{c['end']:.1f}s]: {c['text']}"
        for c in chunks
    ]
    return "\n".join(lines)


def generate_answer(query: str, chunks: list[dict]) -> str:
    """Send query + retrieved context to vLLM and return the answer text."""
    from openai import APIConnectionError, APIStatusError
    client = _build_openai_client()
    try:
        response = client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{_format_context(chunks)}\n\nQuestion: {query}",
                },
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.1,
            **_NO_THINK,
        )
    except APIConnectionError as e:
        raise RuntimeError(f"Cannot reach vLLM at {VLLM_BASE_URL}. Is the server running?") from e
    except APIStatusError as e:
        raise RuntimeError(f"vLLM returned an error: {e.status_code} — {e.message}") from e
    return response.choices[0].message.content.strip()


def generate_answer_stream(query: str, chunks: list[dict]) -> Iterator[str]:
    """Stream answer tokens from vLLM. Yields text delta strings."""
    from openai import APIConnectionError, APIStatusError
    client = _build_openai_client()
    try:
        stream = client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{_format_context(chunks)}\n\nQuestion: {query}",
                },
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.1,
            stream=True,
            **_NO_THINK,
        )
    except APIConnectionError as e:
        raise RuntimeError(f"Cannot reach vLLM at {VLLM_BASE_URL}. Is the server running?") from e
    except APIStatusError as e:
        raise RuntimeError(f"vLLM returned an error: {e.status_code} — {e.message}") from e
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def generate_summary(segments: list[dict]) -> str:
    """Generate a summary of the full transcript using vLLM."""
    from openai import APIConnectionError, APIStatusError
    raw = "\n".join(f"[{s['speaker']} | {s['start']:.1f}s]: {s['text']}" for s in segments)
    transcript = raw[:_SHOW_NOTES_CHAR_BUDGET]
    client = _build_openai_client()
    try:
        response = client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            max_tokens=512,
            temperature=0.3,
            **_NO_THINK,
        )
    except APIConnectionError as e:
        raise RuntimeError(f"Cannot reach vLLM at {VLLM_BASE_URL}. Is the server running?") from e
    except APIStatusError as e:
        raise RuntimeError(f"vLLM returned an error: {e.status_code} — {e.message}") from e
    return response.choices[0].message.content.strip()


def classify_intent(query: str) -> str:
    """Classify query intent using keyword matching. Returns 'summary', 'stats', or 'rag'."""
    q = query.lower()
    if any(k in q for k in [
        "summarize", "summary", "what is this about", "what's this about",
        "what is the podcast about", "overview", "tell me about this",
    ]):
        return "summary"
    if any(k in q for k in [
        "who spoke", "who talks", "speaking time", "talk time", "talked more",
        "spoke more", "how long", "duration", "word count", "how many words", "pace",
    ]):
        return "stats"
    return "rag"


def resolve_speaker_names(segments: list[dict]) -> dict[str, str]:
    """Scan the intro of the transcript to map speaker IDs to real names.

    Returns e.g. {"0": "Lex Fridman", "1": "Elon Musk"}.
    Falls back to {"0": "Speaker 0", ...} if names can't be extracted.
    """
    speakers = sorted({s["speaker"] for s in segments})

    def _readable(spk: str) -> str:
        # "SPEAKER_00" → "Speaker 1", "SPEAKER_01" → "Speaker 2", "0" → "Speaker 1"
        clean = spk.replace("SPEAKER_", "").lstrip("0") or "0"
        try:
            return f"Speaker {int(clean) + 1}"
        except ValueError:
            return f"Speaker {spk}"

    fallback = {spk: _readable(spk) for spk in speakers}

    intro = segments[:25]
    text = "\n".join(f"[{s['speaker']}]: {s['text']}" for s in intro)

    from openai import APIConnectionError, APIStatusError
    client = _build_openai_client()
    try:
        response = client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are given the opening of a podcast transcript with speaker IDs. "
                        "Extract the real name of each speaker ID ONLY if the speaker explicitly introduces themselves or is directly addressed by name in the transcript. "
                        "Do NOT infer or guess names from context, topics, or writing style. "
                        "If a name is not explicitly stated, return null for that speaker. "
                        f"The speaker IDs are: {list(speakers)}. "
                        "Return only a JSON object mapping each exact speaker ID to their name or null, e.g. "
                        f"{{\"{list(speakers)[0]}\": \"Name\"}}. Return only valid JSON, no other text."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=128,
            temperature=0.0,
            **_NO_THINK,
        )
    except (APIConnectionError, APIStatusError) as e:
        logger.warning("speaker name resolution failed (%s) — using fallback labels", e)
        return fallback
    raw = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
        # Replace nulls with fallback, ensure all speakers are covered
        return {
            spk: parsed.get(spk) or fallback[spk]
            for spk in speakers
        }
    except (json.JSONDecodeError, KeyError):
        logger.warning("could not parse speaker name JSON: %r", raw)
        return fallback


def compute_stats(segments: list[dict]) -> dict:
    """Compute episode stats directly from segments — no LLM needed."""
    if not segments:
        return {}

    total_duration = max(s["end"] for s in segments)

    speaker_time: dict[str, float] = {}
    speaker_words: dict[str, int] = {}
    for s in segments:
        spk = s["speaker"]
        speaker_time[spk] = speaker_time.get(spk, 0.0) + (s["end"] - s["start"])
        speaker_words[spk] = speaker_words.get(spk, 0) + len(s["text"].split())

    total_words = sum(speaker_words.values())

    return {
        "duration": round(total_duration),
        "total_words": total_words,
        "avg_pace_wpm": round(total_words / (total_duration / 60)) if total_duration else 0,
        "speakers": [
            {
                "speaker": spk,
                "talk_time": round(speaker_time[spk]),
                "talk_pct": round(speaker_time[spk] / total_duration * 100, 1),
                "words": speaker_words[spk],
            }
            for spk in sorted(speaker_time, key=lambda x: -speaker_time[x])
        ],
    }


def _sample_chunks(chunks: list[dict]) -> list[dict]:
    """Select a representative sample of chunks within the character budget.

    Always keeps the first 3 and last 3 chunks (intro + outro are most
    information-dense in podcasts), then fills remaining budget with evenly
    spaced chunks from the middle.
    """
    if not chunks:
        return []

    def _chars(cs: list[dict]) -> int:
        return sum(len(c["text"]) for c in cs)

    if _chars(chunks) <= _SHOW_NOTES_CHAR_BUDGET:
        return chunks

    n = len(chunks)
    anchors = chunks[:3] + chunks[-3:] if n > 6 else chunks
    anchor_ids = set(list(range(3)) + list(range(max(0, n - 3), n)))
    budget_left = _SHOW_NOTES_CHAR_BUDGET - _chars(anchors)

    middle = [c for i, c in enumerate(chunks) if i not in anchor_ids]
    sampled_middle: list[dict] = []
    if middle and budget_left > 0:
        step = max(1, len(middle) // (budget_left // 300))
        for i in range(0, len(middle), step):
            if _chars(sampled_middle) + len(middle[i]["text"]) > budget_left:
                break
            sampled_middle.append(middle[i])

    combined = sorted(anchors + sampled_middle, key=lambda c: c["start"])
    return combined


def generate_show_notes(chunks: list[dict]) -> dict:
    """Generate timestamped chapters, key quotes, and takeaways from transcript chunks."""
    from openai import APIConnectionError, APIStatusError
    sampled = _sample_chunks(chunks)
    transcript = "\n\n".join(
        f"[{c['start']:.1f}s–{c['end']:.1f}s | {c['speaker']}]\n{c['text']}" for c in sampled
    )
    client = _build_openai_client()
    try:
        response = client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[
                {"role": "system", "content": _SHOW_NOTES_PROMPT},
                {"role": "user", "content": transcript},
            ],
            max_tokens=1024,
            temperature=0.2,
            **_NO_THINK,
        )
    except APIConnectionError as e:
        raise RuntimeError(f"Cannot reach vLLM at {VLLM_BASE_URL}. Is the server running?") from e
    except APIStatusError as e:
        raise RuntimeError(f"vLLM returned an error: {e.status_code} — {e.message}") from e
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.warning("could not parse show notes JSON: %r", raw[:200])
        return {"chapters": [], "quotes": [], "takeaways": [raw]}


def query_rag(
    question: str,
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
) -> tuple[str, list[dict]]:
    """Retrieve relevant chunks then generate an answer. Returns (answer, chunks)."""
    chunks = retrieve(question, collection, embedding_model)
    answer = generate_answer(question, chunks)
    return answer, chunks
