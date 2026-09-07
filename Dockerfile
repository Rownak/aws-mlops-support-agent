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

# Deps layer: the lockfile plus this package's manifest and rag_core's (its
# only workspace dependency). Even with --no-install-project, uv still needs
# to locate every workspace member ON DISK to build its installation plan
# unless the sync is scoped with --package — otherwise it errors trying to
# resolve sibling members (rag_bench_eval, finance_report_rag) whose source
# isn't copied here. Manifests only (no source) keeps this layer cached when
# only code changes.
COPY pyproject.toml uv.lock ./
COPY packages/rag_core/pyproject.toml ./packages/rag_core/
COPY packages/aws_mlops_support_agent/pyproject.toml ./packages/aws_mlops_support_agent/

# --frozen: install exactly uv.lock, error if it disagrees with pyproject.
# --no-dev: skip pytest/ruff. --no-install-project: third-party deps only;
# the workspace members are installed in the next step, once their source is
# present. --package scopes resolution to aws-mlops-support-agent's closure
# (pulls in rag-core transitively) instead of the whole workspace. The cache
# mount persists uv's download cache across builds without ending up in any
# image layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --package aws-mlops-support-agent

# Now the source, and a second sync that installs both workspace members
# themselves. This is what puts the `aws-agent-demo` console script on PATH —
# under the old flat layout the app was run from source, but console scripts
# only exist once the packages are actually installed.
#
# Only copy the two packages this image actually ships — the workspace also
# has rag_bench_eval and finance_report_rag, which pull in benchmark/experiment
# deps (datasets, langchain-ollama, ...) that don't belong in this image.
#
# --no-editable is essential here: uv installs workspace members editable by
# default, which only writes a .pth pointing back at /app/packages. The runtime
# stage copies the venv and NOT the source, so an editable install would leave
# the console script importing a package that isn't there. --no-editable copies
# the code into site-packages, making the venv genuinely self-contained.
#
# --package scopes the sync to aws-mlops-support-agent's dependency closure
# (which pulls in rag-core transitively) instead of the whole workspace —
# otherwise uv sync with no --package installs every member it can see.
COPY packages/rag_core ./packages/rag_core
COPY packages/aws_mlops_support_agent ./packages/aws_mlops_support_agent
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package aws-mlops-support-agent

# ---------- Stage 2: runtime — slim image, venv only ----------
FROM python:3.13-slim-bookworm

# Never run as root in a deployed container: a compromised app process
# shouldn't own the filesystem. UID is pinned to 1000 because Hugging Face
# Spaces runs containers as that specific uid; ECS doesn't care either way.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser

WORKDIR /app

# Both packages are installed INTO the venv (non-editable), so the venv is the
# only thing the runtime needs — no separate source copy, unlike the old layout.
COPY --from=builder /app/.venv ./.venv

# Put the venv first on PATH (so the console scripts resolve) and make logs
# flush immediately — CloudWatch reads stdout line by line.
# STREAMLIT_SERVER_* are Streamlit's own env-var config, so the CMD below
# needs no flags and the port stays overridable per platform (ECS uses the
# 8501 default; Hugging Face Spaces sets STREAMLIT_SERVER_PORT=7860).
# 0.0.0.0: inside a container, localhost is unreachable from the host's
# port mapping. headless: don't try to open a browser server-side.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

USER appuser

EXPOSE 8501

# Streamlit's built-in liveness endpoint. slim has no curl, so use stdlib.
# (Docker-only convenience — ECS task definitions declare their own health
# check and ignore this one.)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f\"http://localhost:{os.environ['STREAMLIT_SERVER_PORT']}/_stcore/health\", timeout=4)" || exit 1

# The console script from aws_mlops_support_agent's [project.scripts]; it
# shims `streamlit run` onto the installed app module, so the container no
# longer needs to know the .py file's path. Server host/port/headless come
# from the STREAMLIT_SERVER_* env vars set above.
CMD ["aws-agent-demo"]
