"""Streamlit UI: upload audio → transcribe → query with speaker-aware RAG."""
import os
import tempfile

import streamlit as st

from pipeline.chunk import chunk_transcript
from pipeline.embed import (
    clear_collection,
    embed_chunks,
    get_chroma_collection,
    get_summary,
    load_embedding_model,
    store_summary,
)
from pipeline.rag import generate_answer_stream, generate_summary, retrieve
from pipeline.transcribe import load_stt_model, transcribe

st.set_page_config(page_title="Audio RAG", layout="wide")


@st.cache_resource
def _stt_model():
    """Cached VibeVoice-ASR processor + model (loaded once per session)."""
    return load_stt_model()


@st.cache_resource
def _embedding_model():
    """Cached sentence-transformers embedding model (loaded once per session)."""
    return load_embedding_model()


def process_audio(audio_path: str, filename: str) -> tuple[list[dict], str]:
    """Transcribe, chunk, summarize, and embed an audio file into ChromaDB.

    Returns (segments, summary).
    """
    collection = get_chroma_collection()

    with st.status("Processing audio…", expanded=True) as status:
        st.write("Transcribing with VibeVoice-ASR…")
        segments = transcribe(audio_path, _stt_model())
        st.write(f"Transcription done — {len(segments)} segments.")

        st.write("Chunking transcript…")
        chunks = chunk_transcript(segments)
        st.write(f"Chunking done — {len(chunks)} chunks.")

        st.write("Generating summary…")
        summary = generate_summary(segments)
        st.write("Summary ready.")

        st.write(f"Ingesting {len(chunks)} chunks into ChromaDB…")
        embed_chunks(chunks, collection, _embedding_model(), audio_filename=filename)
        store_summary(summary, collection, _embedding_model(), audio_filename=filename)
        st.write("Ingestion complete.")

        status.update(label="Done!", state="complete", expanded=False)

    return segments, summary


def render_transcript(segments: list[dict]) -> None:
    """Render transcript segments as a labeled table."""
    st.subheader("Transcript")
    rows = [
        {"Speaker": s["speaker"], "Start (s)": s["start"], "End (s)": s["end"], "Text": s["text"]}
        for s in segments
    ]
    st.dataframe(rows, use_container_width=True)


def render_answer(answer: str, chunks: list[dict]) -> None:
    """Render the RAG answer and expandable source chunks with speaker metadata."""
    with st.expander("Source chunks", expanded=False):
        for i, c in enumerate(chunks, 1):
            st.markdown(
                f"**{i}. {c['speaker']}** &nbsp;|&nbsp; "
                f"{c['start']:.1f}s – {c['end']:.1f}s\n\n{c['text']}"
            )
            st.caption(f"Distance: {c['distance']}  •  File: {c.get('audio_file', '')}")
            st.divider()


def render_sidebar() -> None:
    """Render sidebar controls including the destructive clear button."""
    with st.sidebar:
        st.header("Database")
        st.caption("All uploaded files are indexed and queryable together.")
        st.divider()
        if st.button("Clear all indexed audio", type="secondary"):
            st.session_state["confirm_clear"] = True

        if st.session_state.get("confirm_clear"):
            st.warning("This will delete all indexed audio from ChromaDB.")
            col1, col2 = st.columns(2)
            if col1.button("Yes, clear", type="primary"):
                clear_collection(get_chroma_collection())
                st.session_state.clear()
                st.success("Database cleared.")
                st.rerun()
            if col2.button("Cancel"):
                st.session_state["confirm_clear"] = False
                st.rerun()


def main() -> None:
    """Main Streamlit entry point."""
    render_sidebar()

    st.title("Audio RAG")
    st.caption(
        "Upload audio files — each one is indexed and queryable together. "
        "Previous uploads are preserved across sessions."
    )

    uploaded = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a", "flac"])

    if uploaded is not None:
        if st.session_state.get("last_filename") != uploaded.name:
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            segments, summary = process_audio(tmp_path, uploaded.name)
            os.unlink(tmp_path)

            st.session_state["segments"] = segments
            st.session_state["summary"] = summary
            st.session_state["last_filename"] = uploaded.name

        # If already indexed in a previous session, fetch the stored summary
        elif "summary" not in st.session_state:
            stored = get_summary(get_chroma_collection(), uploaded.name)
            if stored:
                st.session_state["summary"] = stored

    if st.session_state.get("summary"):
        st.subheader("What this podcast is about")
        st.write(st.session_state["summary"])

    if st.session_state.get("segments"):
        render_transcript(st.session_state["segments"])

        st.subheader("Ask a Question")
        query = st.text_input("Enter your question about the audio content")

        if query and st.button("Ask"):
            chunks = retrieve(query, get_chroma_collection(), _embedding_model())
            st.subheader("Answer")
            answer = st.write_stream(generate_answer_stream(query, chunks))
            render_answer(answer, chunks)


if __name__ == "__main__":
    main()
