# Deploying the demo to Streamlit Community Cloud (free)

Free, always-on hosting for the AWS DevOps Support Agent demo. The app is
public — anyone with the link can use it — and capped at 5 questions per
browser session.

Because there is no access gate, **the OpenAI spend cap in step 1 is the only
thing standing between a public URL and your bill.** Do it first, not last.

## Why not Hugging Face Spaces or Vercel

- **Hugging Face Spaces** — as of 2026 the Docker and Gradio SDKs require a
  paid plan (PRO, $9/mo for personal accounts). Only the **Static** SDK is
  free, and Static Spaces serve HTML/CSS/JS from a CDN with no server-side
  Python process. This app cannot run there: it needs a Python runtime for the
  LangGraph state machine and the RAG engine, and the API keys would have to
  sit in client-side JavaScript where any visitor could read them — which is
  far worse than a rate limit, since a leaked key can be drained directly.
- **Vercel** — stateless serverless functions. Streamlit is a stateful
  long-lived WebSocket server, and this app's LangGraph `InMemorySaver`
  (`agent/graph.py`) keeps paused threads in process memory, so a follow-up
  click can land on an instance where the checkpoint doesn't exist.

Streamlit Community Cloud is purpose-built for this app's shape, genuinely
free, and does not sleep aggressively.

## Prerequisites

- The repo pushed to **GitHub** (Community Cloud deploys from a GitHub repo).
- A Pinecone index that is **already populated**. Ingestion is disabled in
  the hosted demo, so run `uv run aws-agent-ingest` locally first.

## 1. Cap the OpenAI key first

The app is public and the per-session limit resets on refresh, so this cap is
your **only** real protection. Do it before the app goes live.

1. Create a **dedicated project** at https://platform.openai.com → Settings →
   Projects. Never reuse a key that has access to your other work.
2. In that project, create a **project-scoped API key** used only by this demo.
3. Set a **hard monthly spend limit** (Settings → Limits) — $5 is plenty for
   `gpt-4o-mini`. Past the cap, requests fail with `429 insufficient_quota`.
4. Add a spend **alert** below the cap. Enforcement isn't instantaneous, so a
   small amount can leak past the limit.

Restricting the key to this one project means the worst case is a capped,
known dollar amount on a key that can't touch anything else.

## 2. What makes this repo deployable

Two files in **`app/`** exist specifically for this host. They must stay
together in that directory — the reason is subtle and is the single most
common way this deploy breaks:

- **`app/streamlit_app.py`** — the entrypoint. Community Cloud does not
  `pip install` this repo's workspace packages (they live under
  `packages/*/src`), so this shim prepends both `src` directories to
  `sys.path` and calls the real app. Local `uv run aws-agent-demo` is
  unaffected.
- **`app/requirements.txt`** — 93 exact pins exported from `uv.lock` with the
  two local packages excluded (`--no-emit-workspace`), since the shim handles
  those.

> **Why `app/` and not the repo root.** Community Cloud uses the *first*
> dependency file it finds, searching the **entrypoint's own directory first**,
> then the repo root, in this order: `uv.lock`, `Pipfile`, `environment.yml`,
> `requirements.txt`, `pyproject.toml`.
>
> This repo has a root `uv.lock`. If the entrypoint lives anywhere that
> doesn't contain its own dependency file, the search falls through to the
> root, finds `uv.lock` first, and tries to build the whole workspace from
> source — including `rag_core`, whose `[tool.uv.sources]` block uses an
> `extra` key that Community Cloud's older `uv` rejects:
>
> ```
> unknown field `extra`, expected one of `git`, `subdirectory`, ...
> help: `rag-core` was included because `aws-mlops-workspace` depends on it
> ```
>
> Keeping the entrypoint and `requirements.txt` together in `app/` means
> `requirements.txt` is found first and the root `uv.lock` is never read.
> Do not move either file out on its own, and do not point **Main file path**
> at `packages/.../demo/streamlit_app.py` — that reintroduces the same failure.

Regenerate `app/requirements.txt` after any dependency change, or the deployed
app silently drifts from `uv.lock`:

```bash
uv export --frozen --no-dev --no-emit-workspace --no-annotate --no-header \
  --package aws-mlops-support-agent --no-hashes -o app/requirements.txt
```

## 3. Deploy

1. Push `deploy_demo` (or merge to `master`) to GitHub.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. **Create app** → **Deploy a public app from GitHub**.
4. Set:
   - **Repository:** your repo
   - **Branch:** the branch you pushed
   - **Main file path:** `app/streamlit_app.py`  ← not the one under
     `packages/`; see the note in step 2
5. Before clicking Deploy, open **Advanced settings** and set Python version
   to **3.13** (this repo requires `>=3.13`).

## 4. Set the secrets

In **Advanced settings** → **Secrets** (or later via app menu → Settings →
Secrets), paste TOML — *not* `KEY=value`:

```toml
OPENAI_API_KEY = "sk-proj-..."
PINECONE_API_KEY = "pcsk_..."
LANGSMITH_API_KEY = "lsv2_..."
```

Community Cloud exposes these to the app as environment variables, which is
exactly what `load_dotenv()` / `os.environ.get` already read — no code change.

Do **not** set `JIRA_*` or `DRY_RUN`. With the Jira vars unset the agent shows
a ticket draft instead of filing one, and the demo forces `dry_run=True`
regardless (`force_dry_run` in `settings.py`).

`OPENAI_CHAT_MODEL` is not needed — `gpt-4o-mini` is already pinned in
`config.yml`.

> Secrets are stored encrypted, but anyone with **write access to the GitHub
> repo** can view them from the Community Cloud dashboard. Keep the repo's
> collaborator list tight.

## 5. Share the link

The app is public: send the URL to HR (or put it on a CV) and it just works —
no sign-in, no password, nothing to explain. Mentioning the 5-question limit
is still worth a line so a reviewer isn't surprised when the input greys out.

## What protects the key

| Layer | Stops | Bypassable? |
|---|---|---|
| OpenAI hard spend cap | Everything, absolutely | **No** |
| 5 runs per session | Casual over-use | **Yes** — F5 resets it |

The per-session counter lives in `st.session_state`, which Streamlit ties to
the WebSocket connection: refreshing the page starts a new session with a
fresh count. It is a courtesy limit and a UX signal, not a security control.

With the app public, **the spend cap is the entire defence**. That is a
reasonable trade for a portfolio demo — the blast radius is a known dollar
amount on a key scoped to one project — but it only holds if step 1 was
actually done. Two things worth doing once it's live:

- Watch the OpenAI usage dashboard for the first few days.
- If it ever gets scraped or abused, the fastest fix is rotating the key in
  Community Cloud's Secrets, not redeploying.

If you later want a real limit, the fix is a server-side counter in
`@st.cache_resource` (shared across sessions, survives refresh) keyed on
something the visitor can't reset.

## Verifying the deploy

1. Open the app URL → the chat UI loads, sidebar showing "5 of 5 questions
   left this session".
2. Ask a question → you get an answer with sources, counter drops to 4.
3. Confirm the sidebar's "Ingest / refresh docs" button is greyed out.
4. Use all 5 → the chat input disables with the limit message.
5. Refresh → the counter resets (expected; see "What protects the key").

## Troubleshooting

**`Failed to build rag-core` / `unknown field 'extra'` / `TOML parse error`.**
Community Cloud read the root `uv.lock` instead of `app/requirements.txt` and
tried to build the workspace from source; `rag_core`'s `[tool.uv.sources]`
uses an `extra` key that its older `uv` doesn't support. Almost always this
means **Main file path** is wrong — it must be `app/streamlit_app.py`. If it
points at `packages/aws_mlops_support_agent/src/.../demo/streamlit_app.py`,
that directory has no dependency file, so the search falls through to the
root `uv.lock`. Fix the path in the app's settings and reboot it.

(Editing `rag_core/pyproject.toml` to drop the `extra` key would also silence
this, but don't — that block is what keeps `rerank-cpu` and `rerank-cuda` from
resolving together, and the deployed image doesn't install either extra
anyway. Fixing the entrypoint path is the correct fix.)

**`ModuleNotFoundError: rag_core`.** The shim didn't run or the paths are
wrong. Confirm **Main file path** is `app/streamlit_app.py`, and that
`app/streamlit_app.py` still computes `_ROOT` as `Path(__file__).parent.parent`
(it must point at the repo root, one level above `app/`).

**App boots but every question errors.** A missing or misnamed secret. Check
the app's logs (bottom-right **Manage app** → logs).

**Resource limits.** Free apps get ~1 GB RAM. This app's runtime deps are
modest (no torch, no markitdown — see the image-slimming work), so it fits,
but avoid re-adding heavy extras to `requirements.txt`.
