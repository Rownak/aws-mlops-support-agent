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

# Deps layer: the lockfile plus EVERY pyproject in the workspace. uv resolves
# the workspace as a whole, so it needs each member's manifest — but not their
# source, which is what keeps this layer cached when only code changes.
COPY pyproject.toml uv.lock ./
COPY packages/rag_core/pyproject.toml ./packages/rag_core/
COPY packages/aws_mlops_support_agent/pyproject.toml ./packages/aws_mlops_support_agent/

# --frozen: install exactly uv.lock, error if it disagrees with pyproject.
# --no-dev: skip pytest/ruff. --no-install-project: third-party deps only;
# the workspace members are installed in the next step, once their source is
# present. The cache mount persists uv's download cache across builds without
# ending up in any image layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Now the source, and a second sync that installs both workspace members
# themselves. This is what puts the `aws-agent-demo` console script on PATH —
# under the old flat layout the app was run from source, but console scripts
# only exist once the packages are actually installed.
#
# --no-editable is essential here: uv installs workspace members editable by
# default, which only writes a .pth pointing back at /app/packages. The runtime
# stage copies the venv and NOT the source, so an editable install would leave
# the console script importing a package that isn't there. --no-editable copies
# the code into site-packages, making the venv genuinely self-contained.
COPY packages ./packages
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ---------- Stage 2: runtime — slim image, venv only ----------
FROM python:3.13-slim-bookworm

# Never run as root in a deployed container: a compromised app process
# shouldn't own the filesystem.
RUN useradd --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

# Both packages are installed INTO the venv (non-editable), so the venv is the
# only thing the runtime needs — no separate source copy, unlike the old layout.
COPY --from=builder /app/.venv ./.venv

# Put the venv first on PATH (so the console scripts resolve) and make logs
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

# The console script from aws_mlops_support_agent's [project.scripts]; it
# shims `streamlit run` onto the installed app module, so the container no
# longer needs to know the .py file's path. Flags after it are passed straight
# through to Streamlit.
# 0.0.0.0: inside a container, localhost is unreachable from the host's port
# mapping. headless: don't try to open a browser server-side.
CMD ["aws-agent-demo", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
