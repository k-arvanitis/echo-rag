"""CLI entry point: transcribe an audio file and optionally query the RAG pipeline.

Usage (after `uv sync`):
    uv run audio-rag path/to/recording.wav
    uv run audio-rag path/to/recording.wav --query "What did speaker A say about X?"
    uv run audio-rag path/to/recording.wav --json
    uv run audio-rag path/to/recording.wav --clear   # wipe DB before indexing

Or directly:
    python run_pipeline.py path/to/recording.wav --query "..."
"""
import argparse
import json

from pipeline.chunk import chunk_transcript
from pipeline.embed import clear_collection, embed_chunks, get_chroma_collection, load_embedding_model
from pipeline.rag import query_rag
from pipeline.transcribe_vibevoice_vllm import load_stt_model, transcribe


def _print_transcript(segments: list[dict]) -> None:
    """Print transcript segments to stdout in a readable format."""
    print("\n=== Transcript ===")
    for s in segments:
        print(f"[{s['speaker']} | {s['start']:.1f}s–{s['end']:.1f}s]  {s['text']}")


def _print_answer(answer: str, chunks: list[dict]) -> None:
    """Print the RAG answer and its source chunks to stdout."""
    print(f"\n=== Answer ===\n{answer}")
    print("\n--- Source chunks ---")
    for i, c in enumerate(chunks, 1):
        print(f"  {i}. [{c['speaker']} | {c['start']:.1f}s–{c['end']:.1f}s]  {c['text']}")
        print(f"     distance: {c['distance']}")


def _index_audio(audio_path: str, clear: bool = False) -> list[dict]:
    """Transcribe, chunk, and embed an audio file. Returns raw segments."""
    print("Loading models…")
    stt = load_stt_model()
    emb = load_embedding_model()
    collection = get_chroma_collection()

    if clear:
        print("Clearing existing ChromaDB data…")
        clear_collection(collection)

    print(f"Transcribing: {audio_path}")
    segments = transcribe(audio_path, stt)

    print(f"Chunking and embedding {len(segments)} segments…")
    chunks = chunk_transcript(segments)
    embed_chunks(chunks, collection, emb, audio_filename=audio_path)
    print(f"Indexed {len(chunks)} chunks into ChromaDB.")

    return segments


def main() -> None:
    """Parse args, run the pipeline, and optionally answer a query."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio and query it with speaker-aware RAG."
    )
    parser.add_argument("audio", help="Path to audio file (wav, mp3, m4a, flac)")
    parser.add_argument(
        "--query", "-q", default=None, help="Question to ask after indexing"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output transcript and answer as JSON"
    )
    parser.add_argument(
        "--clear", action="store_true", help="Wipe ChromaDB before indexing this file"
    )
    args = parser.parse_args()

    segments = _index_audio(args.audio, clear=args.clear)

    if args.json:
        output: dict = {"segments": segments}
        if args.query:
            collection = get_chroma_collection()
            emb = load_embedding_model()
            answer, chunks = query_rag(args.query, collection, emb)
            output["query"] = args.query
            output["answer"] = answer
            output["source_chunks"] = chunks
        print(json.dumps(output, indent=2))
        return

    _print_transcript(segments)

    if args.query:
        collection = get_chroma_collection()
        emb = load_embedding_model()
        answer, chunks = query_rag(args.query, collection, emb)
        _print_answer(answer, chunks)


if __name__ == "__main__":
    main()
