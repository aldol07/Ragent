import os

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
import openai
import streamlit as st

from agent import Agent
from config import EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME, USE_RERANKER
from document_processor import DocumentProcessor
from llm import LLM, LLMNotConfigured
from retriever import HybridRetriever, Models

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")

st.set_page_config(page_title="Ragent", page_icon="📄", initial_sidebar_state="expanded")


# Shared, read-only resources (one copy per server process).
@st.cache_resource(show_spinner="Starting up…")
def get_models():
    return Models(EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME if USE_RERANKER else None)


@st.cache_resource
def get_llm():
    return LLM()


def friendly_error(e):
    if isinstance(e, openai.RateLimitError):
        if "per-day" in str(e) or "per_day" in str(e):
            return "Daily usage limit reached. Please try again tomorrow."
        return "The assistant is busy right now. Please try again in a few seconds."
    if isinstance(e, (openai.APITimeoutError, openai.APIConnectionError)):
        return "Couldn't reach the assistant. Please check your connection and try again."
    return "Something went wrong while answering. Please try again."


def build_index(files):
    chunks, _ = DocumentProcessor().process_files(files)
    if not chunks:
        st.session_state.retriever = None
        st.sidebar.error("No readable text found. Scanned PDFs aren't supported yet.")
        return
    st.session_state.retriever = HybridRetriever(get_models(), chunks)
    st.session_state.messages = []


def render_sources(sources, verification):
    if verification and verification.get("supported") is False and not verification.get("revised"):
        st.caption("⚠️ Some statements could not be confirmed in the documents.")
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for s in sources:
                c = s["chunk"]
                st.markdown(f"**[{s['n']}]** {c.source} · page {c.page}")
                st.caption(c.text[:500] + ("…" if len(c.text) > 500 else ""))


STATUS_LABELS = {
    "search_documents": "Searching documents…",
    "list_documents": "Looking at your files…",
    "calculator": "Calculating…",
}


def main():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("retriever", None)
    st.session_state.setdefault("upload_sig", None)

    st.title("Ragent")
    st.caption("Ask questions about your PDFs and get answers cited to the page.")

    try:
        llm = get_llm()
    except LLMNotConfigured as e:
        st.error(str(e))
        st.stop()
    get_models()  # load once up front so the first upload isn't slow

    # ---------------------------------------------------------------- sidebar
    with st.sidebar:
        st.subheader("Your documents")
        uploads = st.file_uploader(
            "Upload PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
        )
        sig = tuple((f.name, f.size) for f in uploads) if uploads else None
        if sig and sig != st.session_state.upload_sig:
            with st.spinner("Reading your documents…"):
                build_index([(f.name, f.getvalue()) for f in uploads])
            st.session_state.upload_sig = sig

        sample_pdfs = sorted(f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")) if os.path.isdir(DOCS_DIR) else []
        if sample_pdfs and not uploads and st.button("Try with sample documents", use_container_width=True):
            with st.spinner("Reading sample documents…"):
                files = []
                for name in sample_pdfs:
                    with open(os.path.join(DOCS_DIR, name), "rb") as fh:
                        files.append((name, fh.read()))
                build_index(files)
            st.session_state.upload_sig = ("samples",)

        retriever = st.session_state.retriever
        if retriever:
            st.markdown("\n".join(f"- {s['source']} ({s['pages']} pages)" for s in retriever.sources))
            if st.session_state.messages and st.button("New chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    # ------------------------------------------------------------------- chat
    if not retriever:
        st.info("Upload one or more PDFs from the sidebar to get started.")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                render_sources(m.get("sources"), m.get("verification"))

    question = st.chat_input("Ask a question about your documents", disabled=not retriever)
    if not question:
        return

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    agent = Agent(llm, retriever, verify=True)
    with st.chat_message("assistant"):
        progress = st.empty()
        progress.caption("Thinking…")
        box = st.empty()
        text, final = "", None
        try:
            for event in agent.run(question, history):
                kind = event["type"]
                if kind == "token":
                    progress.empty()
                    text += event["text"]
                    box.markdown(text + "▌")
                elif kind == "reset":
                    text = ""
                    box.empty()
                elif kind == "tool":
                    progress.caption(STATUS_LABELS.get(event["name"], "Working…"))
                elif kind == "final":
                    final = event
        except Exception as e:
            print(f"[app] request failed: {e!r}")
            progress.empty()
            box.empty()
            st.error(friendly_error(e))
            st.session_state.messages.pop()  # let the user simply re-ask
            return

        progress.empty()
        box.markdown(final["answer"])
        render_sources(final["sources"], final["verification"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": final["answer"],
        "sources": final["sources"],
        "verification": final["verification"],
    })


main()
