# 📄 DocSensei: PDF and DOCX Q&A

A production-style **RAG (Retrieval-Augmented Generation)** application. Upload PDF or DOCX
documents, ask questions, and get answers grounded strictly in your content — with inline
citations, zero hallucination, and a built-in experiment comparing two chunking strategies.

---

## Features

- 📤 **Multi-document upload** — PDF and DOCX, one or many at once
- ✂️ **Two chunking strategies** — Recursive Character Splitting and Sentence-Based Splitting, switchable from the sidebar
- 🔬 **Built-in chunking comparison** — runs both strategies on the same document/question and reports chunk count, avg. chunk size, retrieval latency, top similarity score, and generated answer, with an automatic recommendation
- 🧠 **Configurable LLM provider** — Google Gemini, OpenAI, or Groq (pick one, only that provider's API key is needed)
- 📌 **Inline citations** — every answer cites `(DocumentName, Page X)` or `(DocumentName, Chunk Y)`
- 🚫 **No hallucination** — strict prompt forces the model to say *"I do not know based on the provided document."* when the answer isn't present
- 💬 **Conversation memory** — ask follow-up questions without re-uploading
- 🔍 **Search inside retrieved chunks** — filter the source panel by keyword
- 📊 **Document statistics** — pages, chunks, tokens, embedding model, upload time
- 🗑️ **Clear Knowledge Base** button — wipes the persisted ChromaDB store
- ⬇️ **Downloads** — chat history (JSON) and the chunking comparison report (Markdown)
- ⚡ **Streaming responses** (toggle in sidebar)
- 🛡️ **Graceful error handling** — corrupted PDFs, empty files, missing API keys, empty questions, and vector DB errors are all caught with a clear message instead of a crash

---

## Architecture

```
User uploads PDF/DOCX
        │
        ▼
  loader.py  → extracts text + page metadata
        │
        ▼
 splitter.py → chunks text (Recursive OR Sentence-based)
        │
        ▼
embeddings.py → all-MiniLM-L6-v2 (local, free, no API key)
        │
        ▼
vectorstore.py → stores chunks + vectors in ChromaDB (persisted to disk)
        │
        ▼
   User asks a question
        │
        ▼
retriever.py → embeds the question, finds top-k similar chunks
        │
        ▼
  prompt.py  → builds a strict "answer ONLY from this context" prompt
        │
        ▼
   rag.py    → sends prompt to the chosen LLM (Gemini / OpenAI / Groq)
        │
        ▼
  Answer + citations + source chunks shown in app.py (Streamlit UI)
```

## Project Structure

```
DocSensei/
├── app.py                  # Streamlit UI, wires everything together
├── requirements.txt
├── .env.example
├── README.md
├── utils/
│   ├── loader.py            # PDF/DOCX reading
│   ├── splitter.py           # Both chunking strategies
│   ├── embeddings.py          # Sentence-Transformers wrapper
│   ├── vectorstore.py          # ChromaDB build / persist / clear
│   ├── retriever.py             # Top-k similarity search
│   ├── prompt.py                 # Strict citation-enforcing prompt
│   ├── rag.py                     # Multi-provider RAG pipeline
│   ├── comparison.py                # Chunking strategy comparison engine
│   └── helpers.py                    # Token counts, validation, timers
├── data/                     # (scratch space, empty by default)
├── chroma_db/                # persisted vector store (created at runtime)
├── assets/                   # (for screenshots etc.)
└── tests/
    └── test_basic.py          # Unit tests (chunking logic, validation)
```

---

## How Retrieval Works

1. Your document is split into overlapping chunks so ideas aren't cut off mid-sentence.
2. Each chunk is converted into a 384-number vector using the free, local `all-MiniLM-L6-v2` model — no API key needed for this step, it runs on your CPU in seconds.
3. Those vectors are stored in **ChromaDB**, a local vector database, persisted to disk under `chroma_db/`.
4. When you ask a question, it's embedded the same way, and the chunks with the closest vectors (by cosine similarity) are retrieved.
5. Only the LLM generation step (Gemini/OpenAI/Groq) requires an API key and internet access.

## Chunking Comparison (Mandatory Outcome)

The **🔬 Chunking Comparison** tab runs both strategies — **Recursive Character Splitter** and
**Sentence-Based Splitter** — on your uploaded document, using the same test question, and reports:

| Metric | What it tells you |
|---|---|
| Number of chunks | How granular each strategy is |
| Avg. chunk size | Character length per chunk |
| Retrieval latency | How fast similarity search runs |
| Top similarity score | How relevant the best-matched chunk is |
| Generated answer | Side-by-side answer quality |

It then recommends a winner (weighted toward relevance, with latency as a tie-breaker) and lets you
export the full report as Markdown.

---

## Installation

```bash
# 1. Clone / unzip the project, then enter the folder
cd DocSensei

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows PowerShell
# or: source venv/Scripts/activate    # Git Bash on Windows
# or: source venv/bin/activate        # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
copy .env.example .env            # Windows
# or: cp .env.example .env         # macOS/Linux/Git Bash
# then open .env and paste in ONE provider's key (Gemini, OpenAI, or Groq)

# 5. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. You can also paste the API key directly into the
sidebar instead of using `.env` — either works.

## Environment Variables

| Variable | Provider | Where to get it |
|---|---|---|
| `GOOGLE_API_KEY` | Google Gemini | https://aistudio.google.com/apikey (free tier available) |
| `OPENAI_API_KEY` | OpenAI | https://platform.openai.com/api-keys |
| `GROQ_API_KEY` | Groq | https://console.groq.com/keys (free tier available) |

You only need **one** — set the matching provider in the sidebar dropdown.

## Screenshots

_Add your own screenshots here after running the app locally:_
- `assets/chat_view.png` — main chat interface with citations
- `assets/comparison_view.png` — chunking comparison side-by-side
- `assets/stats_view.png` — document statistics panel

## Running Tests

```bash
python -m pytest tests/
# or
python -m unittest tests.test_basic
```

These tests cover the pure-logic pieces (sentence splitting, chunk factory, upload validation)
and don't require an API key or internet access.

## Deployment

### Streamlit Community Cloud
1. Push this project to a public/private GitHub repo.
2. Go to https://share.streamlit.io → "New app" → select the repo and `app.py`.
3. Under "Secrets", add your API key, e.g.:
   ```toml
   GOOGLE_API_KEY = "your-key-here"
   ```
4. Deploy. Note: Streamlit Cloud's filesystem is ephemeral, so the ChromaDB persisted index
   resets on redeploy — that's expected for a demo app.

### Render / Railway
1. Push to GitHub.
2. Create a new **Web Service** and point it at the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Add your API key as an environment variable in the service's dashboard.

## Future Improvements

- Add a reranking step (e.g. cross-encoder) after initial retrieval for higher precision.
- Support scanned/image-only PDFs via OCR (e.g. `pytesseract`).
- Add user authentication for multi-user deployments.
- Swap ChromaDB's local persistence for a hosted vector DB for cloud deployments where disk isn't persistent.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` on startup | Run `pip install -r requirements.txt` again inside the activated venv |
| "No API key provided" error | Paste a key into the sidebar, or fill in `.env` and restart the app |
| First question is slow | The embedding model downloads once (~90MB) on first run, then it's cached |
| `chromadb` fails to build on Windows | Upgrade pip first: `python -m pip install --upgrade pip`, then reinstall |
| PDF returns "could not be read" | The PDF may be scanned images with no real text layer — OCR isn't included yet |
| Answers seem to ignore new uploads | Click "Clear Knowledge Base" in the sidebar to force a rebuild |
| Streaming looks like it "answers twice" | This is expected — the app streams the answer, then quietly re-runs once to fetch source citations reliably. Turn off "Enable streaming responses" if you'd rather see one pass |

---
