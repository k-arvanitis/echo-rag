"""Retrieve relevant chunks and generate answers via OpenAI API."""
import json
import logging
import re
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from config import (
    HYDE_ENABLED,
    LLM_MODEL,
    MAX_TOKENS,
    OPENAI_API_KEY,
    RERANKER_ENABLED,
    RERANKER_MODEL,
    TOP_K_RESULTS,
)

# Candidates fetched from ChromaDB before reranking.
# Needs to be larger than TOP_K_RESULTS so the reranker has room to reorder.
_RERANKER_FETCH_K = 10

# Module-level reranker instance — loaded once on first retrieval call.
_reranker: Any = None

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
    """Create an OpenAI client using the configured API key."""
    return OpenAI(api_key=OPENAI_API_KEY)

# ~4500 tokens; leaves room for system prompt + response within 8k context
_SHOW_NOTES_CHAR_BUDGET = 18_000


def _get_reranker() -> Any:
    """Lazily load and cache the cross-encoder reranker.

    Returns the CrossEncoder instance, or None if reranking is disabled or
    the model fails to load (in which case a warning is logged).
    """
    global _reranker
    if not RERANKER_ENABLED:
        return None
    if _reranker is not None:
        return _reranker
    try:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.info("reranker loaded: %s", RERANKER_MODEL)
    except Exception as e:
        logger.warning("reranker failed to load (%s) — falling back to embedding-distance order", e)
        _reranker = None
    return _reranker


def _generate_hyde_query(query: str) -> str:
    """Generate a hypothetical transcript excerpt for HyDE retrieval.

    Sends the query to the vLLM server asking for a 1-2 sentence hypothetical
    answer as if it were spoken in a meeting transcript. Returns the hypothetical
    text to embed in place of the raw query.

    Falls back to the original query string if the vLLM call fails.
    """
    from openai import APIConnectionError, APIStatusError
    client = _build_openai_client()
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are helping retrieve relevant meeting transcript segments. "
                        "Write a 1-2 sentence hypothetical transcript excerpt that would answer this question:"
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=100,
            temperature=0.5,

        )
        hypothetical = response.choices[0].message.content.strip()
        logger.debug("HyDE hypothetical: %r", hypothetical)
        return hypothetical
    except (APIConnectionError, APIStatusError) as e:
        logger.warning("HyDE generation failed (%s) — falling back to direct query embedding", e)
        return query


def retrieve(
    query: str,
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    top_k: int = TOP_K_RESULTS,
) -> list[dict]:
    """Embed query and return top-k chunks, with optional reranking.

    Pipeline:
      1. HyDE (if enabled): generate a hypothetical transcript excerpt and embed
         that instead of the raw query to improve recall on vague questions.
      2. ChromaDB ANN search: fetch `_RERANKER_FETCH_K` candidates (more than
         top_k so the reranker has candidates to reorder).
      3. Cross-encoder reranking (if enabled): re-score every (query, chunk) pair
         and sort by reranker score.  Falls back to embedding-distance order if the
         model failed to load.
    """
    # Step 1: optionally replace query with a hypothetical document for HyDE
    embed_text = _generate_hyde_query(query) if HYDE_ENABLED else query
    query_embedding = embedding_model.encode(embed_text).tolist()

    # Step 2: fetch candidates — more than top_k when reranking is on
    fetch_k = max(_RERANKER_FETCH_K, top_k * 2) if RERANKER_ENABLED else top_k + 2

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
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

    # Step 3: rerank if a cross-encoder is available
    reranker = _get_reranker()
    if reranker is not None and len(chunks) > top_k:
        try:
            pairs = [(query, c["text"]) for c in chunks]
            scores = reranker.predict(pairs)
            chunks = [c for _, c in sorted(zip(scores, chunks), key=lambda x: -x[0])]
            logger.debug("reranked %d candidates → top %d", len(chunks), top_k)
        except Exception as e:
            logger.warning("reranking failed (%s) — using embedding-distance order", e)

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
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{_format_context(chunks)}\n\nQuestion: {query}",
                },
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.1,

        )
    except APIConnectionError as e:
        raise RuntimeError(f"Cannot reach OpenAI API. Check your OPENAI_API_KEY and network connection.") from e
    except APIStatusError as e:
        raise RuntimeError(f"OpenAI API error: {e.status_code} — {e.message}") from e
    return response.choices[0].message.content.strip()


def generate_answer_stream(query: str, chunks: list[dict]) -> Iterator[str]:
    """Stream answer tokens from vLLM. Yields text delta strings."""
    from openai import APIConnectionError, APIStatusError
    client = _build_openai_client()
    try:
        stream = client.chat.completions.create(
            model=LLM_MODEL,
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

        )
    except APIConnectionError as e:
        raise RuntimeError(f"Cannot reach OpenAI API. Check your OPENAI_API_KEY and network connection.") from e
    except APIStatusError as e:
        raise RuntimeError(f"OpenAI API error: {e.status_code} — {e.message}") from e
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
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            max_tokens=512,
            temperature=0.3,

        )
    except APIConnectionError as e:
        raise RuntimeError(f"Cannot reach OpenAI API. Check your OPENAI_API_KEY and network connection.") from e
    except APIStatusError as e:
        raise RuntimeError(f"OpenAI API error: {e.status_code} — {e.message}") from e
    return response.choices[0].message.content.strip()


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
            model=LLM_MODEL,
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
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _SHOW_NOTES_PROMPT},
                {"role": "user", "content": transcript},
            ],
            max_tokens=1024,
            temperature=0.2,

        )
    except APIConnectionError as e:
        raise RuntimeError(f"Cannot reach OpenAI API. Check your OPENAI_API_KEY and network connection.") from e
    except APIStatusError as e:
        raise RuntimeError(f"OpenAI API error: {e.status_code} — {e.message}") from e
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
