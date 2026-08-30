"""Manual LIVE check for the Jira wrapper — creates a REAL Jira ticket.

NOT a pytest test (deliberately not named test_*.py so pytest never
collects it — a live API call must never run inside the offline suite).

Prerequisites: JIRA_BASE_URL / JIRA_EMAIL / JIRA_API_TOKEN /
JIRA_PROJECT_KEY filled in .env (see .env.example for setup steps).

Run with:  uv run python packages/aws_mlops_support_agent/tests/jira_live_check.py

dry_run is overridden for this one call only — .env stays DRY_RUN=true,
so nothing else in the app can accidentally go live.
"""

from dataclasses import replace

from aws_mlops_support_agent.agent.jira_tool import create_issue
from aws_mlops_support_agent.agent.ticket import TicketDraft
from aws_mlops_support_agent.settings import load_settings


def main() -> None:
    cfg = replace(load_settings(), dry_run=False)  # live for THIS call only

    draft = TicketDraft(
        summary="Live test — jira_tool",
        description=("Manual verification that create_issue posts a real ticket. Safe to close."),
        docs_checked=[],
        suggested_next_steps=[],
    )

    result = create_issue(draft, cfg)
    print("Created:", result["key"])
    print("View it:", f"{cfg.jira_base_url}/browse/{result['key']}")


if __name__ == "__main__":
    main()
