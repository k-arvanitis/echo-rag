"""Embed speaker-turn chunks and store them in ChromaDB (HTTP server mode)."""
import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_HOST, CHROMA_PORT, EMBEDDING_MODEL

COLLECTION_NAME: str = "audio_rag"


def get_chroma_collection() -> chromadb.Collection:
    """Connect to the ChromaDB server and return (or create) the collection."""
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return client.get_or_create_collection(COLLECTION_NAME)


def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Load the sentence-transformers embedding model."""
    return SentenceTransformer(model_name)


def embed_chunks(
    chunks: list[dict],
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    audio_filename: str = "",
) -> None:
    """Embed chunks and upsert into Chroma, preserving speaker metadata."""
    texts = [c["text"] for c in chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=False).tolist()

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

    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)


def store_summary(
    summary: str,
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    audio_filename: str,
) -> None:
    """Embed and upsert the summary as a special document for direct lookup."""
    embedding = embedding_model.encode(summary, show_progress_bar=False).tolist()
    collection.upsert(
        ids=[f"{audio_filename}_summary"],
        embeddings=[embedding],
        documents=[summary],
        metadatas=[{
            "type": "summary",
            "audio_file": audio_filename,
            "speaker": "",
            "start": 0.0,
            "end": 0.0,
        }],
    )


def get_summary(collection: chromadb.Collection, audio_filename: str) -> str | None:
    """Retrieve a previously stored summary for an audio file. Returns None if missing."""
    result = collection.get(ids=[f"{audio_filename}_summary"])
    if result["documents"]:
        return result["documents"][0]
    return None


def clear_collection(collection: chromadb.Collection) -> None:
    """Delete all documents from the collection (called explicitly, never automatically)."""
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
