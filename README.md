# Ragent — Agentic RAG over your PDFs

Upload PDFs and ask questions. Ragent answers with **page-level citations**, using a
tool-calling research agent over a **hybrid retriever**, and a second **verifier agent**
that fact-checks every answer against the retrieved passages before you see it.

Live demo: [ragent-07.streamlit.app](https://ragent-07.streamlit.app/)

## How it works

```text
            ┌──────────────────────────── Researcher agent (LLM via OpenRouter) ────────┐
question ──►│  plans → calls tools → reads passages → streams a cited answer            │
            │     tools: search_documents(query, source?) · list_documents · calculator │
            └───────────────┬───────────────────────────────────────────────────────────┘
                            │ search_documents
                            ▼
         ┌──────────────── Hybrid retriever (per-session index) ────────────────┐
         │ dense: MiniLM embeddings + FAISS (cosine)  ┐                         │
         │ sparse: BM25                               ┴► RRF fusion ► cross-   │
         │                                                encoder rerank ► top-k│
         └──────────────────────────────────────────────────────────────────────┘
                            │ draft answer + evidence
                            ▼
              Verifier agent: are all claims supported?
                 yes → answer      no → rewrite using evidence only
```

- **Ingestion** — `pypdf` text extraction page by page, recursive chunking (800 chars, 150 overlap); every chunk keeps its file name and page number for citations.
- **Hybrid retrieval** — dense search catches paraphrases, BM25 catches exact terms (IDs, names, numbers). Results are merged with Reciprocal Rank Fusion and reranked by a cross-encoder (`ms-marco-MiniLM-L-6-v2`).
- **Research agent** — an LLM tool-calling loop (max 5 steps). It splits multi-part questions into several searches, rewrites follow-ups into standalone queries using chat history, can restrict a search to one file, and uses an AST-whitelisted calculator instead of doing math itself.
- **Verifier agent** — checks each claim in the draft against the cited passages; unsupported claims are removed by a rewrite pass. The UI shows whether an answer was verified or corrected.
- **Clean streaming UI** — minimal Streamlit chat: answers stream in with a short progress hint ("Searching documents…"), and every answer has an expandable source list (file, page, passage).
- **Isolation** — each browser session gets its own index; only the read-only models are shared across users.

## Quick start

```bash
git clone https://github.com/aldol07/Ragent.git
cd Ragent
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env           # then put your OpenRouter key in .env
streamlit run app.py
```

Open http://localhost:8501, upload PDFs in the sidebar, and ask away. (Locally, any PDFs you drop into a `docs/` folder appear behind a **Try with sample documents** button.)

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | — (required) | Key from openrouter.ai |
| `LLM_MODEL` | `nvidia/nemotron-3-super-120b-a12b:free` | Any OpenRouter model that supports tool calling (free by default) |
| `LLM_FALLBACK_MODELS` | `qwen/qwen3.8-27b:free,google/gemma-4-31b-it:free` | Tried in order if the primary is rate-limited or down |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Dense embeddings |
| `RERANKER_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `USE_RERANKER` | `true` | Turn reranking off for faster CPU-only runs |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | Chunking, in characters |
| `TOP_K_RESULTS` | `5` | Passages returned per search |
| `MAX_AGENT_STEPS` | `5` | Upper limit on researcher tool-calling rounds |

On Streamlit Cloud, put the same keys in the app's **Secrets**.

## Project structure

```text
app.py                 Streamlit chat UI
agent.py               Researcher (tool-calling loop) + verifier/reviser agents
retriever.py           Dense + BM25 + RRF + cross-encoder hybrid retriever
document_processor.py  PDF → page-aware chunks
tools.py               Tool schemas and the safe calculator
llm.py                 OpenRouter client (streaming, tool-call assembly, token tracking)
config.py              Settings from environment
```

## Evaluation

**Retrieval** was measured on 48 hand-labelled questions over 5 PDFs (48 chunks): product FAQs
plus two technical notes on a physics-informed neural network. Questions are paraphrased rather than
copied from the text, and each is labelled with every chunk that contains its answer.

| Retriever | Hit@1 | Recall@5 | MRR | Page Recall@5 | Latency / query (CPU, warm) |
|---|---|---|---|---|---|
| Dense (MiniLM + FAISS) | 0.542 | 0.875 | 0.675 | 0.875 | ~20 ms |
| BM25 | 0.625 | 0.875 | 0.727 | 0.896 | <1 ms |
| Hybrid (RRF) | 0.583 | 0.896 | 0.719 | 0.917 | ~120 ms |
| **Hybrid + cross-encoder rerank** | **0.792** | **0.938** | **0.838** | **0.958** | ~750 ms |

Re-ranking the fused candidates is the biggest single gain: the top result is the right passage for
79% of questions, against 54% for dense search alone (+25 points), and the right page is in the top 5
for 96%. Fusion on its own mainly improves recall; the cross-encoder fixes the ordering.

Limitations: the corpus is small, so these numbers show the relative effect of each stage rather than
absolute performance on large collections. End-to-end answer quality (LLM-judged
correctness and how often the verifier corrects a draft) is still to be measured.
