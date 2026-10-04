import os

from dotenv import load_dotenv

load_dotenv()


def _flag(name, default):
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# LLM (OpenRouter, OpenAI-compatible API)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
# Tried in order when the primary model is rate-limited or down (free models often are).
LLM_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("LLM_FALLBACK_MODELS", "qwen/qwen3.8-27b:free,google/gemma-4-31b-it:free").split(",")
    if m.strip()
]
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4000"))  # reasoning models spend tokens thinking

# Embeddings and reranking
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
USE_RERANKER = _flag("USE_RERANKER", True)

# Chunking (characters)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Retrieval
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
CANDIDATE_POOL = int(os.getenv("CANDIDATE_POOL", "20"))  # per retriever, before fusion/rerank
RRF_K = 60

# Agent
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "5"))
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "4"))
