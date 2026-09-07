"""Entrypoint for Streamlit Community Cloud.  Set "Main file path" to
`app/streamlit_app.py`.

Two problems this file solves, both specific to Community Cloud:

1. It installs the pins in requirements.txt but does NOT `pip install` this
   repo's two workspace packages (rag_core, aws_mlops_support_agent), which
   live under packages/*/src — so neither is importable. The sys.path lines
   below fix that.

2. Community Cloud picks the FIRST dependency file it finds, searching the
   entrypoint's own directory before the repo root, in this order:
   uv.lock, Pipfile, environment.yml, requirements.txt, pyproject.toml.
   The repo root has a uv.lock, which outranks requirements.txt and drags in
   the whole workspace — including rag_core's `tool.uv.sources` block, whose
   `extra` key Community Cloud's older uv rejects outright.

   Keeping this file and requirements.txt together in app/ means that
   requirements.txt is found first and the root uv.lock is never read.
   That is the whole reason this lives in app/ rather than the repo root —
   don't move either file out on its own.

Locally nothing changes: `uv run aws-agent-demo` still uses the console script.
"""

import sys
from pathlib import Path

# app/ -> repo root
_ROOT = Path(__file__).parent.parent
for _pkg in ("rag_core", "aws_mlops_support_agent"):
    _src = _ROOT / "packages" / _pkg / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from aws_mlops_support_agent.demo.streamlit_app import main  # noqa: E402

main()
