"""Streamlit UI: upload audio → transcribe → query with speaker-aware RAG."""
import os
import tempfile

import streamlit as st

from pipeline.chunk import chunk_transcript
from pipeline.embed import clear_collection, embed_chunks, get_chroma_collection, load_embedding_model
from pipeline.rag import query_rag
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


def process_audio(audio_path: str, filename: str) -> list[dict]:
    """Transcribe, chunk, and embed an audio file — accumulates into ChromaDB."""
    with st.spinner("Transcribing with VibeVoice-ASR…"):
        segments = transcribe(audio_path, _stt_model())

    with st.spinner("Chunking and embedding into ChromaDB…"):
        chunks = chunk_transcript(segments)
        embed_chunks(chunks, get_chroma_collection(), _embedding_model(), audio_filename=filename)

    return segments


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
    st.subheader("Answer")
    st.write(answer)

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

            segments = process_audio(tmp_path, uploaded.name)
            os.unlink(tmp_path)

            st.session_state["segments"] = segments
            st.session_state["last_filename"] = uploaded.name

    if st.session_state.get("segments"):
        render_transcript(st.session_state["segments"])

        st.subheader("Ask a Question")
        query = st.text_input("Enter your question about the audio content")

        if query and st.button("Ask"):
            with st.spinner("Retrieving and generating answer…"):
                answer, chunks = query_rag(query, get_chroma_collection(), _embedding_model())
            render_answer(answer, chunks)


if __name__ == "__main__":
    main()
