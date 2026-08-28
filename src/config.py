"""Application configuration via environment variables with pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env into os.environ before anything else reads it.
#
# pydantic-settings already reads .env for RAG_*-prefixed fields, but only into
# the Settings object below — it never populates os.environ. The OpenAI and
# Anthropic SDKs read OPENAI_API_KEY / ANTHROPIC_API_KEY straight from
# os.environ, so without this call a key set in .env (exactly as the README
# instructs) is invisible to them and every generation call fails with
# "Missing credentials".
#
# The path is resolved relative to this file rather than the cwd so the key is
# found regardless of where the process was launched from. override=False keeps
# real environment variables winning over .env, which is the precedence
# containers and CI expect.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)


class Settings(BaseSettings):
    """Central configuration for the RAG pipeline.

    All values can be overridden via environment variables or a .env file.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- Paths ----------
    data_dir: Path = Path("data")
    chroma_path: Path = Path("data/chroma_db")
    chroma_host: str | None = None
    chroma_port: int | None = None
    golden_dataset_path: Path = Path("data/golden_dataset/dataset.jsonl")
    eval_results_path: Path = Path("data/eval_results")

    # ---------- Embedding model ----------
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384
    # LRU cache size for repeated/paraphrased query embeddings (0 disables caching).
    embedding_query_cache_size: int = 256

    # ---------- Chunking ----------
    chunk_size: int = 800
    chunk_overlap: int = 150

    # ---------- Ingestion limits ----------
    max_file_size_mb: float = 50.0
    max_pdf_pages: int = 1000

    # ---------- Generation limits ----------
    # Caps the total characters of retrieved context sent to the LLM, as a
    # cost/runaway-prompt guardrail.
    max_context_chars: int = 12000

    # ---------- Retrieval ----------
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    top_k_final: int = 5

    # Hybrid search (Phase 2)
    hybrid_alpha: float = 0.6  # weight for vector search (1.0 = pure vector, 0.0 = pure BM25)
    rrf_k: int = 60  # constant for Reciprocal Rank Fusion

    # ---------- Re-ranker (Phase 2) ----------
    reranker_model: str = "BAAI/bge-reranker-large"

    # ---------- LLM ----------
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # ---------- Evaluation (Phase 3) ----------
    faithfulness_threshold: float = 0.7
    eval_llm_model: str = "gpt-4o-mini"

    # ---------- Logging ----------
    log_level: str = "INFO"

    # ---------- API / CLI ----------
    api_url: str = "http://localhost:8000"


def _build_settings() -> Settings:
    """Construct Settings, reporting bad config without echoing its value.

    Pydantic's ValidationError embeds the offending input in its message. When
    the offending input is a mis-pasted credential — e.g. an API key landing in
    RAG_LLM_PROVIDER, which only accepts "openai"/"anthropic" — the default
    error writes that secret straight into stdout, logs, and any crash reporter
    downstream. Name the fields at fault, never their contents.
    """
    try:
        return Settings()
    except ValidationError as exc:
        fields = sorted(
            {".".join(str(part) for part in err["loc"]) or "(root)" for err in exc.errors()}
        )
        env_names = ", ".join(f"RAG_{name.upper()}" for name in fields)
        raise RuntimeError(
            f"Invalid RAG configuration for: {env_names}. "
            "Check these variables in your .env file or environment. "
            "(Offending values are omitted so secrets are never written to logs.)"
        ) from None


settings = _build_settings()
