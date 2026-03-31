"""Retrieve relevant chunks and generate answers via vLLM."""
from collections.abc import Iterator

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from config import MAX_TOKENS, TOP_K_RESULTS, VLLM_BASE_URL, VLLM_MODEL

_SYSTEM_PROMPT = (
    "You are an assistant that answers questions about audio transcripts. "
    "Use only the provided transcript context. "
    "Reference speakers by their label (e.g. SPEAKER_00) when relevant."
)

_SUMMARY_SYSTEM_PROMPT = (
    "You are given the full transcript of a podcast or audio recording. "
    "Write a concise 2-3 paragraph summary covering the main topics discussed, "
    "key points made, and who said what where relevant. "
    "Be specific and informative — avoid generic statements."
)


def _build_openai_client() -> OpenAI:
    """Create an OpenAI client pointed at the local vLLM server."""
    return OpenAI(base_url=VLLM_BASE_URL, api_key="dummy")


def retrieve(
    query: str,
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    top_k: int = TOP_K_RESULTS,
) -> list[dict]:
    """Embed query and return top-k chunks with their speaker metadata."""
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({**meta, "text": doc, "distance": round(dist, 4)})
    return chunks


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a labelled context block for the prompt."""
    lines = [
        f"[{c['speaker']} | {c['start']:.1f}s–{c['end']:.1f}s]: {c['text']}"
        for c in chunks
    ]
    return "\n".join(lines)


def generate_answer(query: str, chunks: list[dict]) -> str:
    """Send query + retrieved context to vLLM and return the answer text."""
    client = _build_openai_client()
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
    )
    return response.choices[0].message.content.strip()


def generate_answer_stream(query: str, chunks: list[dict]) -> Iterator[str]:
    """Stream answer tokens from vLLM. Yields text delta strings."""
    client = _build_openai_client()
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
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def generate_summary(segments: list[dict]) -> str:
    """Generate a summary of the full transcript using vLLM."""
    transcript = "\n".join(
        f"[{s['speaker']} | {s['start']:.1f}s]: {s['text']}" for s in segments
    )
    client = _build_openai_client()
    response = client.chat.completions.create(
        model=VLLM_MODEL,
        messages=[
            {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        max_tokens=512,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def query_rag(
    question: str,
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
) -> tuple[str, list[dict]]:
    """Retrieve relevant chunks then generate an answer. Returns (answer, chunks)."""
    chunks = retrieve(question, collection, embedding_model)
    answer = generate_answer(question, chunks)
    return answer, chunks
