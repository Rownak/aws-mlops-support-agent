"""Task 5.2 — CloudWatch-friendly structured logging: one JSON object per line.

Why this shape: on ECS Fargate the `awslogs` driver ships each stdout line to
CloudWatch as one log event, and CloudWatch Logs auto-parses JSON lines so
you can filter with expressions like `{ $.event = "escalated" }` — no log
agent or extra dependency needed.

The rest of the codebase uses exactly one function:

    log_event("retrieve_done", k=4, top_score=0.62)

Events are silent until `setup_json_logging()` is called (once, at app
startup) — so tests and library-style imports produce no log output, and the
human-facing `print`s in the CLI stay the primary UX. Logs are the machine
stream, prints are the human stream.
"""

import json
import logging
import sys
from datetime import UTC, datetime

# One named logger for the whole agent; not the root logger, so third-party
# libraries' log records don't get sprayed into our JSON stream.
_LOGGER_NAME = "agent"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        # `log_event` smuggles its keyword fields in via `extra=`, which
        # attaches them as a `fields` attribute on the record.
        entry.update(getattr(record, "fields", {}))
        # default=str: a non-JSON-native field (dataclass, Path, ...) must
        # never crash logging — degrade to its string form instead.
        return json.dumps(entry, default=str)


def log_event(event: str, **fields) -> None:
    """Emit one structured log line, e.g. log_event("user_action", action="retry")."""
    logging.getLogger(_LOGGER_NAME).info(event, extra={"fields": fields})


def setup_json_logging(level: int = logging.INFO) -> None:
    """Route agent events to stdout as JSON lines. Call once at startup."""
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()  # idempotent: calling twice must not double-log
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    # Don't ALSO bubble up to the root logger (which would print the message
    # a second time in its own format if the app ever configures root).
    logger.propagate = False
