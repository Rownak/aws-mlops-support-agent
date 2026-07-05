# ---------- Stage 1: builder — install locked deps into a venv ----------
# Official uv image = python:3.13-slim-bookworm + the uv binary. Same Debian
# release as the runtime stage below, so the venv's symlink to the system
# python resolves identically after the copy.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

# Compile .pyc at build time (faster container start), copy instead of
# hardlink across the cache mount boundary.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Deps layer: only the two files that define dependencies. Editing src/ never
# invalidates this layer, so rebuilds skip the slow install entirely.
COPY pyproject.toml uv.lock ./

# --frozen: install exactly uv.lock, error if it disagrees with pyproject.
# --no-dev: skip pytest/ruff. --no-install-project: deps only (src/ isn't
# copied yet, and the app is run from source, not installed as a package).
# The cache mount persists uv's download cache across builds without ending
# up in any image layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ---------- Stage 2: runtime — slim image, venv + source only ----------
FROM python:3.13-slim-bookworm

# Never run as root in a deployed container: a compromised app process
# shouldn't own the filesystem.
RUN useradd --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY --from=builder /app/.venv ./.venv
COPY src ./src

# Put the venv first on PATH (so `streamlit` resolves to it) and make logs
# flush immediately — CloudWatch reads stdout line by line.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

USER appuser

EXPOSE 8501

# Streamlit's built-in liveness endpoint. slim has no curl, so use stdlib.
# (Docker-only convenience — ECS task definitions declare their own health
# check and ignore this one.)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)" || exit 1

# 0.0.0.0: inside a container, localhost is unreachable from the host's
# port mapping. headless: don't try to open a browser server-side.
CMD ["streamlit", "run", "src/demo/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
