"""Task 4.1 — Thin Jira REST wrapper.

Standalone and stateless: takes a `TicketDraft` + `Config`, returns a dict.
Not wired into the graph yet — Phase 4.2 calls this from the `escalate`
node. Kept separate so it's testable and runnable on its own against a
free Jira Cloud instance.
"""

import requests

from src.agent.ticket import TicketDraft
from src.config import Config

JIRA_REQUIRED_FIELDS = ("jira_base_url", "jira_email", "jira_api_token", "jira_project_key")


def _build_payload(draft: TicketDraft, project_key: str) -> dict:
    return {
        "fields": {
            "project": {"key": project_key},
            "summary": draft.summary,
            # Jira Cloud v3 requires the description in Atlassian Document
            # Format (ADF), not a plain string — this is the minimal ADF
            # shape for a single paragraph of text.
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": draft.description}],
                    }
                ],
            },
            "issuetype": {"name": "Task"},
        }
    }


def create_issue(draft: TicketDraft, cfg: Config) -> dict:
    """Create (or, in dry-run, log) a Jira issue from a ticket draft.

    Always validates config first, regardless of dry_run, so a misconfigured
    .env is caught immediately rather than only when someone flips DRY_RUN
    off. Dry-run is the safety default (see config.py) — real API calls only
    happen when DRY_RUN is explicitly false.
    """
    missing = [name for name in JIRA_REQUIRED_FIELDS if not getattr(cfg, name)]
    if missing:
        raise RuntimeError(f"Missing required Jira config: {', '.join(missing)}.")

    payload = _build_payload(draft, cfg.jira_project_key)
    # rstrip: tolerate a trailing slash in JIRA_BASE_URL (…atlassian.net/).
    url = f"{cfg.jira_base_url.rstrip('/')}/rest/api/3/issue"

    if cfg.dry_run:
        print(f"\n=== DRY_RUN: would POST to {url} ===")
        print(payload)
        return {"dry_run": True, "url": url, "payload": payload}

    # Jira Cloud API tokens authenticate via HTTP Basic auth as
    # (account email, token) — this is Atlassian's documented scheme, not a
    # generic bearer token.
    response = requests.post(
        url,
        json=payload,
        auth=(cfg.jira_email, cfg.jira_api_token),
        timeout=10,
    )
    if not response.ok:
        # Jira's 4xx bodies name the exact offending field (e.g. bad project
        # key or issue type) — without this, all you'd see is "400 Bad
        # Request", which is undebuggable.
        raise RuntimeError(f"Jira returned {response.status_code}: {response.text}")
    return response.json()
