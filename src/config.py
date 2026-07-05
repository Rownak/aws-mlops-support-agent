"""Load and validate configuration from environment variables.

All keys come from the environment — locally via a gitignored .env file,
in prod from AWS Secrets Manager. Call `load_config()` wherever settings
are needed; it fails fast with a clear message if a required var is missing.

Verify manually with:  uv run python -m src.config
"""

import os
from dataclasses import dataclass, fields

from dotenv import load_dotenv

# The app can't do anything useful without these two. Jira vars are NOT
# required here — they're only needed in Phase 4, and the Jira tool
# validates them itself when it's actually used.
REQUIRED_VARS = ["OPENAI_API_KEY", "PINECONE_API_KEY"]


@dataclass(frozen=True)
class Config:
    # OpenAI (LLM + embeddings). The embedding model must be identical at
    # ingestion time and query time, or retrieval silently degrades —
    # that's why it lives here and not hardcoded in the ingest script.
    openai_api_key: str
    openai_chat_model: str
    openai_embedding_model: str
    # Pinecone (vector DB)
    pinecone_api_key: str
    pinecone_index_name: str
    # AWS
    aws_region: str
    # Jira (optional until Phase 4)
    jira_base_url: str | None
    jira_email: str | None
    jira_api_token: str | None
    jira_project_key: str | None
    # Safety gate on real Jira ticket creation. Defaults to True.
    dry_run: bool


def _parse_dry_run(value: str | None) -> bool:
    """Fail-safe: only an explicit false/0/no turns dry-run OFF.

    Unset, empty, or a typo like "flase" all stay True, so a config
    mistake can never cause real Jira tickets to be created.
    """
    if value is None:
        return True
    return value.strip().lower() not in ("false", "0", "no")


def load_config() -> Config:
    # Reads .env into os.environ; a no-op if the file is absent (e.g. in CI,
    # where vars come from the environment directly). Does not override
    # vars that are already set.
    load_dotenv()

    # Collect ALL missing vars before failing, so the user fixes them in
    # one pass instead of replaying error-by-error.
    # (Empty string counts as missing — .env.example ships blank values.)
    missing = [name for name in REQUIRED_VARS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in."
        )

    # `os.environ.get(X) or default` (not `.get(X, default)`) so that an
    # empty string in .env still falls back to the default.
    return Config(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_chat_model=os.environ.get("OPENAI_CHAT_MODEL") or "gpt-4o-mini",
        openai_embedding_model=os.environ.get("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small",
        pinecone_api_key=os.environ["PINECONE_API_KEY"],
        pinecone_index_name=os.environ.get("PINECONE_INDEX_NAME") or "aws-mlops-docs",
        aws_region=os.environ.get("AWS_REGION") or "us-east-1",
        jira_base_url=os.environ.get("JIRA_BASE_URL") or None,
        jira_email=os.environ.get("JIRA_EMAIL") or None,
        jira_api_token=os.environ.get("JIRA_API_TOKEN") or None,
        jira_project_key=os.environ.get("JIRA_PROJECT_KEY") or None,
        dry_run=_parse_dry_run(os.environ.get("DRY_RUN")),
    )


if __name__ == "__main__":
    # Manual sanity check. Secrets are never printed — only whether they're set.
    SECRET_FIELDS = {"openai_api_key", "pinecone_api_key", "jira_api_token"}
    cfg = load_config()
    for f in fields(cfg):
        value = getattr(cfg, f.name)
        if f.name in SECRET_FIELDS:
            value = "(set)" if value else "(not set)"
        print(f"{f.name} = {value}")
