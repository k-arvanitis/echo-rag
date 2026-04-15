"""Streamlit UI: upload audio → transcribe → query with speaker-aware RAG."""
import os
import tempfile
import uuid

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import streamlit as st

from pipeline.chunk import chunk_transcript
from pipeline.embed import (
    clear_collection,
    embed_chunks,
    get_chroma_collection,
    get_show_notes,
    get_summary,
    load_embedding_model,
    store_show_notes,
    store_summary,
)
from pipeline.rag import (
    generate_answer_stream,
    generate_show_notes,
    generate_summary,
    resolve_speaker_names,
    retrieve,
)
from pipeline.transcribe_vibevoice_vllm import load_stt_model, transcribe

st.set_page_config(page_title="Echo", layout="wide", page_icon="🎙️")

# Hide Streamlit chrome
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def _stt_model():
    return load_stt_model()


@st.cache_resource(show_spinner=False)
def _embedding_model():
    return load_embedding_model()


def _collection():
    return get_chroma_collection(st.session_state["user_id"])


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def process_audio(audio_path: str, filename: str) -> tuple[list[dict], str, dict] | None:
    """Run the full pipeline. Returns (segments, summary, show_notes) or None on failure."""
    try:
        collection = _collection()
    except RuntimeError as e:
        st.error(str(e))
        return None

    with st.status("Processing audio…", expanded=True) as status:
        try:
            st.write("Transcribing with VibeVoice-ASR…")
            segments = transcribe(audio_path, _stt_model())
            st.write(f"Transcription done — {len(segments)} segments.")
        except Exception as e:
            status.update(label="Transcription failed", state="error")
            st.error(f"Transcription error: {e}")
            return None

        try:
            st.write("Identifying speakers…")
            name_map = resolve_speaker_names(segments)
            for s in segments:
                s["speaker"] = name_map.get(s["speaker"], s["speaker"])
            st.write(f"Speakers: {', '.join(name_map.values())}")
        except Exception as e:
            st.warning(f"Speaker name resolution skipped: {e}")
            name_map = {}

        st.write("Chunking transcript…")
        chunks = chunk_transcript(segments)
        st.write(f"Chunking done — {len(chunks)} chunks.")

        try:
            st.write("Generating summary…")
            summary = generate_summary(segments)
            st.write("Summary ready.")
        except RuntimeError as e:
            status.update(label="Failed", state="error")
            st.error(str(e))
            return None

        try:
            st.write("Generating show notes…")
            show_notes = generate_show_notes(chunks)
            st.write(f"Show notes ready — {len(show_notes.get('chapters', []))} chapters.")
        except RuntimeError as e:
            st.warning(f"Show notes skipped: {e}")
            show_notes = {}

        try:
            st.write(f"Ingesting {len(chunks)} chunks into ChromaDB…")
            embed_chunks(chunks, collection, _embedding_model(), audio_filename=filename)
            store_summary(summary, collection, _embedding_model(), audio_filename=filename)
            store_show_notes(show_notes, collection, audio_filename=filename)
            st.write("Ingestion complete.")
        except RuntimeError as e:
            status.update(label="Ingestion failed", state="error")
            st.error(str(e))
            return None

        status.update(label="Done!", state="complete", expanded=False)

    return segments, summary, show_notes


def render_show_notes(show_notes: dict) -> None:
    chapters = show_notes.get("chapters", [])
    quotes = show_notes.get("quotes", [])
    takeaways = show_notes.get("takeaways", [])

    if chapters:
        st.subheader("Chapters")
        for ch in chapters:
            with st.container(border=True):
                st.markdown(f"**{_fmt_ts(ch.get('start', 0))}** — {ch.get('title', '')}")
                if ch.get("summary"):
                    st.caption(ch["summary"])

    if takeaways:
        st.subheader("Key Takeaways")
        with st.container(border=True):
            for t in takeaways:
                st.markdown(f"- {t}")

    if quotes:
        st.subheader("Notable Quotes")
        for q in quotes:
            with st.container(border=True):
                st.markdown(f"> {q.get('text', '')}")
                st.caption(f"{q.get('speaker', '')} · {_fmt_ts(q.get('start', 0))}")


def render_transcript(segments: list[dict]) -> None:
    def _speaker_label(s: dict) -> str:
        text = s["text"].strip()
        return "-" if text.startswith("[") and text.endswith("]") else s["speaker"]

    rows = [
        {"Speaker": _speaker_label(s), "Start": _fmt_ts(s["start"]), "End": _fmt_ts(s["end"]), "Text": s["text"]}
        for s in segments
    ]
    st.dataframe(rows, use_container_width=True)


def render_answer(answer: str, chunks: list[dict]) -> None:
    audio_bytes = st.session_state.get("audio_bytes")
    with st.expander("Sources", expanded=False):
        for i, c in enumerate(chunks, 1):
            with st.container(border=True):
                st.markdown(
                    f"**{i}. {c['speaker']}** &nbsp;|&nbsp; "
                    f"{_fmt_ts(c['start'])} – {_fmt_ts(c['end'])}\n\n{c['text']}"
                )
                if audio_bytes:
                    st.audio(audio_bytes, start_time=int(c["start"]))
                st.caption(f"File: {c.get('audio_file', '')}  •  Distance: {c['distance']}")


def render_sidebar() -> None:
    """Render sidebar widgets. Returns uploaded file."""
    with st.sidebar:
        st.title("🎙️ Echo")
        st.caption("Query your meetings, calls, and interviews.")
        st.divider()

        uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "flac"])

        st.divider()
        if st.button("Clear my indexed audio", type="secondary", use_container_width=True):
            st.session_state["confirm_clear"] = True

        if st.session_state.get("confirm_clear"):
            st.warning("This will delete all your indexed audio.")
            col1, col2 = st.columns(2)
            if col1.button("Yes, clear", type="primary"):
                clear_collection(_collection())
                for key in ["segments", "summary", "show_notes", "last_filename", "audio_bytes", "confirm_clear"]:
                    st.session_state.pop(key, None)
                st.success("Cleared.")
                st.rerun()
            if col2.button("Cancel"):
                st.session_state["confirm_clear"] = False
                st.rerun()

    return uploaded


def main() -> None:
    # Assign a unique ID per browser session for collection isolation
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = str(uuid.uuid4())

    uploaded = render_sidebar()

    if uploaded is not None:
        if st.session_state.get("last_filename") != uploaded.name:
            suffix = os.path.splitext(uploaded.name)[1]
            audio_bytes = uploaded.read()
            st.session_state["audio_bytes"] = audio_bytes
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            result = process_audio(tmp_path, uploaded.name)
            os.unlink(tmp_path)

            if result is None:
                return  # error already shown by process_audio

            segments, summary, show_notes = result
            st.session_state["segments"] = segments
            st.session_state["summary"] = summary
            st.session_state["show_notes"] = show_notes
            st.session_state["last_filename"] = uploaded.name

        elif "summary" not in st.session_state:
            collection = _collection()
            st.session_state["summary"] = get_summary(collection, uploaded.name)
            st.session_state["show_notes"] = get_show_notes(collection, uploaded.name)

    if not st.session_state.get("summary"):
        st.markdown("## Upload a meeting recording to get started")
        st.caption("Supports MP3, WAV, M4A, FLAC · Transcription + speaker diarization runs locally")
        return

    tab_overview, tab_show_notes, tab_transcript, tab_ask = st.tabs(
        ["Overview", "Show Notes", "Transcript", "Ask"]
    )

    with tab_overview:
        st.subheader(st.session_state.get("last_filename", "Episode"))
        with st.container(border=True):
            st.write(st.session_state["summary"])

    with tab_show_notes:
        show_notes = st.session_state.get("show_notes") or {}
        if show_notes:
            render_show_notes(show_notes)
        else:
            st.info("Show notes not available for this episode.")

    with tab_transcript:
        if st.session_state.get("segments"):
            render_transcript(st.session_state["segments"])
        else:
            st.info("Transcript not available — re-upload the file to generate it.")

    with tab_ask:
        with st.form("ask_form", border=False):
            query = st.text_input("Ask anything about this recording", placeholder="What did we commit to? What objections came up?")
            submitted = st.form_submit_button("Ask", type="primary")

        if submitted and query:
            try:
                chunks = retrieve(query, _collection(), _embedding_model())
                with st.container(border=True):
                    answer = st.write_stream(generate_answer_stream(query, chunks))
                render_answer(answer, chunks)
            except RuntimeError as e:
                st.error(str(e))


if __name__ == "__main__":
    main()
