"""Task 5.2 — offline tests for the JSON logging helper."""

import json
import logging

from src.observability import log_event, setup_json_logging


def test_log_event_emits_one_json_line_on_stdout(capsys):
    setup_json_logging()
    log_event("retrieve_done", k=4, top_score=0.62, is_confident=True)

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["event"] == "retrieve_done"
    assert entry["level"] == "INFO"
    assert entry["k"] == 4
    assert entry["top_score"] == 0.62
    assert entry["is_confident"] is True
    assert "ts" in entry


def test_setup_twice_does_not_double_log(capsys):
    setup_json_logging()
    setup_json_logging()
    log_event("user_action", action="retry")
    assert len(capsys.readouterr().out.strip().splitlines()) == 1


def test_non_json_native_field_degrades_to_string(capsys):
    # A field logging can't serialize must never crash the agent.
    setup_json_logging()
    log_event("escalated", where=object())
    entry = json.loads(capsys.readouterr().out)
    assert entry["event"] == "escalated"
    assert isinstance(entry["where"], str)


def test_silent_until_setup(capsys):
    # Library-style imports and tests that never call setup must stay quiet.
    logger = logging.getLogger("agent")
    logger.handlers.clear()
    logger.propagate = False  # keep pytest's root handler out of the picture
    log_event("question_received", question="q")
    assert capsys.readouterr().out == ""
