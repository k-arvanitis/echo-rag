"""Embed speaker-turn chunks and store them in ChromaDB (HTTP server mode)."""
import json
import logging

import chromadb

logger = logging.getLogger(__name__)
from sentence_transformers import SentenceTransformer

from config import CHROMA_HOST, CHROMA_PORT, EMBEDDING_MODEL

_SHOW_NOTES_PLACEHOLDER = [0.0] * 1024

def get_chroma_collection(user_id: str = "default") -> chromadb.Collection:
    """Connect to the ChromaDB server and return (or create) a per-user collection."""
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        return client.get_or_create_collection(f"audio_rag_{user_id}")
    except Exception as e:
        raise RuntimeError(
            f"Cannot connect to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}. "
            "Is the server running? Try: docker compose up -d chroma"
        ) from e


def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Load the sentence-transformers embedding model."""
    return SentenceTransformer(model_name)


def is_audio_indexed(collection: chromadb.Collection, audio_filename: str) -> bool:
    """Return whether transcript chunks already exist for the given audio file."""
    result = collection.get(where={"audio_file": audio_filename})
    return any(
        metadata.get("type") not in {"summary", "show_notes"}
        for metadata in result.get("metadatas", [])
    )



def embed_chunks(
    chunks: list[dict],
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    audio_filename: str = "",
) -> None:
    """Embed chunks and upsert into Chroma, preserving speaker metadata."""
    texts = [c["text"] for c in chunks]
    ids = [f"{audio_filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "speaker": c["speaker"],
            "start": c["start"],
            "end": c["end"],
            "audio_file": audio_filename,
        }
        for c in chunks
    ]

    embeddings = embedding_model.encode(texts, show_progress_bar=False).tolist()
    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    logger.info("upserted %d chunks for %s", len(chunks), audio_filename)


def _metadata(audio_filename: str, doc_type: str) -> dict:
    return {
        "type": doc_type,
        "audio_file": audio_filename,
        "speaker": "",
        "start": 0.0,
        "end": 0.0,
    }



def _store_document(
    *,
    collection: chromadb.Collection,
    doc_id: str,
    document: str,
    metadata: dict,
    embedding: list[float],
) -> None:
    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[document],
        metadatas=[metadata],
    )



def _get_document(collection: chromadb.Collection, doc_id: str) -> str | None:
    result = collection.get(ids=[doc_id])
    if result["documents"]:
        return result["documents"][0]
    return None



def store_summary(
    summary: str,
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    audio_filename: str,
) -> None:
    """Embed and upsert the summary as a special document for direct lookup."""
    embedding = embedding_model.encode(summary, show_progress_bar=False).tolist()
    _store_document(
        collection=collection,
        doc_id=f"{audio_filename}_summary",
        document=summary,
        metadata=_metadata(audio_filename, "summary"),
        embedding=embedding,
    )



def get_summary(collection: chromadb.Collection, audio_filename: str) -> str | None:
    """Retrieve a previously stored summary for an audio file. Returns None if missing."""
    return _get_document(collection, f"{audio_filename}_summary")



def store_show_notes(
    show_notes: dict,
    collection: chromadb.Collection,
    audio_filename: str,
) -> None:
    """Store show notes JSON as a non-embedded document (direct lookup only)."""
    _store_document(
        collection=collection,
        doc_id=f"{audio_filename}_show_notes",
        document=json.dumps(show_notes),
        metadata=_metadata(audio_filename, "show_notes"),
        embedding=_SHOW_NOTES_PLACEHOLDER,
    )



def get_show_notes(collection: chromadb.Collection, audio_filename: str) -> dict | None:
    """Retrieve stored show notes for an audio file. Returns None if missing."""
    document = _get_document(collection, f"{audio_filename}_show_notes")
    if document is None:
        return None
    try:
        return json.loads(document)
    except json.JSONDecodeError:
        return None


def clear_collection(collection: chromadb.Collection) -> None:
    """Delete all documents from the collection (called explicitly, never automatically)."""
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
