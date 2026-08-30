"""Console-script launcher for the Streamlit demo:  uv run aws-agent-demo

Streamlit is a script runner, not a library entrypoint — `streamlit run` needs
a FILE path. So this shim resolves the installed location of streamlit_app.py
and hands it to Streamlit's own CLI, which keeps the command short and
independent of where the package happens to be installed.
"""

import sys
from pathlib import Path

from streamlit.web import cli as streamlit_cli

APP_PATH = Path(__file__).parent / "streamlit_app.py"


def main() -> None:
    # streamlit's CLI reads sys.argv, so we rewrite it as if the user had
    # typed `streamlit run <path>` and pass through any extra flags.
    sys.argv = ["streamlit", "run", str(APP_PATH), *sys.argv[1:]]
    streamlit_cli.main()


if __name__ == "__main__":
    main()
