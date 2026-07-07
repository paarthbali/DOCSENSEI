"""
DocSensei: PDF and DOCX Q&A
===========================

A production-style RAG (Retrieval-Augmented Generation) application.
Upload PDF/DOCX documents, ask questions, and get answers grounded
strictly in the uploaded content, with inline citations.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from utils.embeddings import EmbeddingManager, DEFAULT_EMBEDDING_MODEL, AVAILABLE_EMBEDDING_MODELS
from utils.helpers import approx_token_count, format_timestamp
from utils.loader import DocumentLoader
from utils.splitter import get_splitter, attach_chunk_metadata, ALL_STRATEGIES, STRATEGY_RECURSIVE
from utils.vectorstore import VectorStoreManager
from utils.rag import RAGPipeline, get_llm, MissingAPIKeyError, ALL_PROVIDERS, DEFAULT_MODELS, PROVIDER_GEMINI
from utils.comparison import ChunkingComparison

load_dotenv()

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(page_title="DocSensei — PDF/DOCX Q&A", page_icon="📄", layout="wide")

st.title("📄 DocSensei: PDF and DOCX Q&A")
st.caption(
    "Grounded document Q&A with inline citations — no hallucinations. "
    "Answers come only from what you upload."
)

# --------------------------------------------------------------------------
# Session state defaults
# --------------------------------------------------------------------------
DEFAULTS = {
    "chat_history": [],
    "vectordb": None,
    "raw_docs": None,
    "doc_stats": None,
    "last_signature": None,
    "comparison_results": None,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# --------------------------------------------------------------------------
# Sidebar — configuration
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    provider = st.selectbox("LLM Provider", ALL_PROVIDERS, index=0)
    env_key_map = {
        PROVIDER_GEMINI: "GOOGLE_API_KEY",
        "OpenAI": "OPENAI_API_KEY",
        "Groq": "GROQ_API_KEY",
    }
    env_var = env_key_map.get(provider, "GOOGLE_API_KEY")
    api_key = st.text_input(
        f"{provider} API Key",
        type="password",
        value=os.getenv(env_var, ""),
        help=f"Falls back to the {env_var} environment variable if left blank.",
    )
    model_name = st.text_input("Model name", value=DEFAULT_MODELS.get(provider, ""))

    st.divider()
    st.subheader("📎 Embedding Model")
    embedding_model = st.selectbox("Model", AVAILABLE_EMBEDDING_MODELS, index=0)

    st.divider()
    st.subheader("✂️ Chunking")
    strategy = st.radio("Strategy", ALL_STRATEGIES, index=0)
    chunk_size = st.slider("Chunk size (characters)", 200, 2000, 800, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 100, step=50)
    top_k = st.slider("Chunks retrieved per answer (top-k)", 1, 10, 4)

    st.divider()
    streaming_enabled = st.checkbox("Enable streaming responses", value=True)

    st.divider()
    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        vm = VectorStoreManager()
        vm.clear_all()
        for key in DEFAULTS:
            st.session_state[key] = DEFAULTS[key]
        st.success("Knowledge base cleared.")
        st.rerun()

    if st.session_state["chat_history"]:
        st.download_button(
            "⬇️ Download chat history",
            data=json.dumps(st.session_state["chat_history"], indent=2),
            file_name="docsensei_chat_history.json",
            mime="application/json",
            use_container_width=True,
        )


@st.cache_resource(show_spinner="Loading embedding model...")
def cached_embeddings(model_name: str):
    return EmbeddingManager(model_name=model_name)


def build_signature(files, strategy, chunk_size, chunk_overlap, embedding_model) -> tuple:
    return (
        tuple(sorted(f.name for f in files)),
        strategy,
        chunk_size,
        chunk_overlap,
        embedding_model,
    )


def process_uploads(files):
    """Load, chunk, embed, and index uploaded files. Returns doc stats dict."""
    loaded_docs, load_errors = DocumentLoader.load_multiple(files)
    for err in load_errors:
        st.warning(f"⚠️ {err}")

    if not loaded_docs:
        raise ValueError("No documents could be loaded.")

    all_raw_docs = []
    for ld in loaded_docs:
        all_raw_docs.extend(ld.documents)

    embed_manager = cached_embeddings(embedding_model)
    splitter = get_splitter(strategy, chunk_size, chunk_overlap)
    chunks = splitter.split_documents(all_raw_docs)
    chunks = attach_chunk_metadata(chunks, strategy)

    vm = VectorStoreManager()
    collection_name = "main_kb"
    vm.clear_collection(collection_name)  # rebuild fresh each time inputs change
    vectordb = vm.build(chunks, embed_manager.embeddings, collection_name)

    total_tokens = sum(approx_token_count(d.page_content) for d in all_raw_docs)
    stats = {
        "files": [ld.filename for ld in loaded_docs],
        "num_pages": sum(ld.num_pages for ld in loaded_docs),
        "num_chunks": len(chunks),
        "strategy": strategy,
        "embedding_model": embedding_model,
        "total_tokens_approx": total_tokens,
        "upload_time": format_timestamp(),
    }

    st.session_state["vectordb"] = vectordb
    st.session_state["raw_docs"] = all_raw_docs
    st.session_state["doc_stats"] = stats
    return stats


# --------------------------------------------------------------------------
# Main tabs
# --------------------------------------------------------------------------
tab_chat, tab_compare, tab_stats = st.tabs(["💬 Chat", "🔬 Chunking Comparison", "📊 Document Stats"])

uploaded_files = st.file_uploader(
    "Upload one or more PDF/DOCX files",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    key="uploader",
)

if uploaded_files:
    signature = build_signature(uploaded_files, strategy, chunk_size, chunk_overlap, embedding_model)
    if st.session_state["last_signature"] != signature:
        try:
            with st.spinner("Reading, chunking, and embedding your documents..."):
                process_uploads(uploaded_files)
            st.session_state["last_signature"] = signature
            st.toast("Knowledge base ready ✅")
        except ValueError as exc:
            st.error(f"❌ {exc}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"❌ Unexpected error while processing documents: {exc}")

# ---------------- TAB 1: Chat ----------------
with tab_chat:
    if not st.session_state["vectordb"]:
        st.info("Upload a document above to start asking questions.")
    else:
        # Render chat history
        for turn in st.session_state["chat_history"]:
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                st.write(turn["answer"])
                if turn.get("sources"):
                    with st.expander("📌 Sources"):
                        for s in turn["sources"]:
                            st.markdown(f"**{s['citation']}** (score: {s.get('score', 'N/A')})")
                            st.caption(s["preview"])

        question = st.chat_input("Ask a question about your document(s)...")

        if question:
            try:
                if not question.strip():
                    raise ValueError("Question cannot be empty.")

                llm = get_llm(provider, model_name, api_key)
                rag = RAGPipeline(llm)

                with st.chat_message("user"):
                    st.write(question)

                with st.chat_message("assistant"):
                    if streaming_enabled:
                        full_answer = st.write_stream(
                            rag.stream_answer(
                                st.session_state["vectordb"],
                                question,
                                top_k=top_k,
                                chat_history=st.session_state["chat_history"],
                            )
                        )
                        # Re-run a non-streaming call to reliably capture sources
                        # (streaming generator above already produced the text).
                        result = rag.answer(
                            st.session_state["vectordb"], question, top_k=top_k,
                            chat_history=st.session_state["chat_history"],
                        )
                        sources = result.sources
                        answer_text = full_answer or result.answer
                    else:
                        result = rag.answer(
                            st.session_state["vectordb"], question, top_k=top_k,
                            chat_history=st.session_state["chat_history"],
                        )
                        answer_text = result.answer
                        sources = result.sources
                        st.write(answer_text)

                    source_records = []
                    if sources:
                        with st.expander("📌 Sources"):
                            search_term = st.text_input(
                                "🔍 Search inside retrieved chunks", key=f"search_{len(st.session_state['chat_history'])}"
                            )
                            for doc in sources:
                                citation = doc.metadata.get("citation", "Unknown")
                                preview = doc.page_content[:500]
                                if search_term and search_term.lower() not in preview.lower():
                                    continue
                                st.markdown(f"**{citation}**")
                                st.caption(preview + ("..." if len(doc.page_content) > 500 else ""))
                                source_records.append({"citation": citation, "preview": preview})

                st.session_state["chat_history"].append(
                    {"question": question, "answer": answer_text, "sources": source_records,
                     "timestamp": format_timestamp()}
                )

            except MissingAPIKeyError as exc:
                st.error(f"🔑 {exc}")
            except ValueError as exc:
                st.error(f"❌ {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ Something went wrong generating the answer: {exc}")

# ---------------- TAB 2: Chunking Comparison ----------------
with tab_compare:
    st.write(
        "Compares **Recursive Character Splitting** vs. **Sentence-Based Splitting** "
        "on your uploaded document(s), using the same question, and reports which "
        "strategy retrieves more relevant, higher-quality context."
    )

    if not st.session_state.get("raw_docs"):
        st.info("Upload a document above first.")
    else:
        compare_question = st.text_input("Test question for comparison:", key="compare_question")
        if st.button("▶️ Run Comparison", disabled=not compare_question):
            try:
                llm = get_llm(provider, model_name, api_key)
                embed_manager = cached_embeddings(embedding_model)
                vm = VectorStoreManager()
                comparator = ChunkingComparison(vm)

                with st.spinner("Running both chunking strategies..."):
                    results = comparator.run(
                        raw_docs=st.session_state["raw_docs"],
                        embeddings=embed_manager.embeddings,
                        llm=llm,
                        question=compare_question,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        top_k=top_k,
                    )
                st.session_state["comparison_results"] = (results, compare_question)
            except MissingAPIKeyError as exc:
                st.error(f"🔑 {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ Comparison failed: {exc}")

        if st.session_state.get("comparison_results"):
            results, q = st.session_state["comparison_results"]
            cols = st.columns(len(results))
            for col, r in zip(cols, results):
                with col:
                    st.subheader(r.strategy)
                    st.metric("Chunks", r.num_chunks)
                    st.metric("Avg. chunk size", f"{r.avg_chunk_size} chars")
                    st.metric("Retrieval latency", f"{r.retrieval_latency_seconds}s")
                    st.metric("Top similarity score", r.top_similarity_score)
                    with st.expander("Generated answer"):
                        st.write(r.answer)
                    with st.expander("Retrieved chunk previews"):
                        for p in r.retrieved_previews:
                            st.caption(p)

            winner = ChunkingComparison.recommend(results)
            st.success(f"👉 Recommended strategy for this document: **{winner}**")

            report_md = ChunkingComparison.to_markdown_report(results, q)
            st.download_button(
                "⬇️ Export comparison report (Markdown)",
                data=report_md,
                file_name="chunking_comparison_report.md",
                mime="text/markdown",
            )

# ---------------- TAB 3: Document Stats ----------------
with tab_stats:
    stats = st.session_state.get("doc_stats")
    if not stats:
        st.info("Upload a document above to see statistics.")
    else:
        st.subheader("📊 Knowledge Base Statistics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Files indexed", len(stats["files"]))
        c1.metric("Total pages", stats["num_pages"])
        c2.metric("Total chunks", stats["num_chunks"])
        c2.metric("Approx. tokens", stats["total_tokens_approx"])
        c3.metric("Chunking strategy", stats["strategy"])
        c3.metric("Embedding model", stats["embedding_model"].split("/")[-1])

        st.markdown("**Files:**")
        for f in stats["files"]:
            st.write(f"- {f}")
        st.caption(f"Last indexed: {stats['upload_time']}")
