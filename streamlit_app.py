"""Root entrypoint for Streamlit Community Cloud.

Community Cloud runs `streamlit run <entrypoint>` against a plain checkout —
it installs the pins in requirements.txt but does NOT `pip install` this
repo's two workspace packages (rag_core, aws_mlops_support_agent), which
live under packages/*/src. So neither is importable by default.

Rather than restructure the repo for one host, this shim prepends both src
directories to sys.path and hands off to the real app. Locally nothing
changes: `uv run aws-agent-demo` still uses the installed console script.

Keep this file at the repo root — Community Cloud resolves the entrypoint
and its dependency file relative to the repo.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent
for _pkg in ("rag_core", "aws_mlops_support_agent"):
    _src = _ROOT / "packages" / _pkg / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from aws_mlops_support_agent.demo.streamlit_app import main  # noqa: E402

main()
