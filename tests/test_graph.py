"""End-to-end graph tests (tasks 3.2–3.5) with injected fakes — no network.

These exercise the real LangGraph machinery: checkpointing, the interrupt
pause/resume cycle, conditional edges, and the retry loop cap.
"""

import pytest
from langgraph.types import Command

from src.agent.graph import MAX_ATTEMPTS, build_graph
from src.agent.state import initial_state
from src.config import Config
from src.rag.retriever import RetrievedChunk


def _cfg():
    """Config stub — nodes never touch it when fakes are injected."""
    return Config(
        openai_api_key="fake",
        openai_chat_model="fake-model",
        openai_embedding_model="fake-embed",
        pinecone_api_key="fake",
        pinecone_index_name="fake-index",
        aws_region="us-east-1",
        jira_base_url=None,
        jira_email=None,
        jira_api_token=None,
        jira_project_key=None,
        dry_run=True,
    )


def _chunk(score):
    return RetrievedChunk(
        text="Docs about env vars.",
        score=score,
        service="codebuild",
        source_file="f.md",
        heading="Env vars",
        url="https://docs.aws/a.html",
    )


class FakeRetriever:
    def __init__(self, score):
        self.score = score
        self.k_values = []

    def __call__(self, question, k=4):
        self.k_values.append(k)
        return [_chunk(self.score), _chunk(self.score - 0.05)]


def fake_answerer(question, chunks):
    return f"Fake answer to: {question}"


def _build(score=0.6):
    retriever = FakeRetriever(score)
    graph = build_graph(_cfg(), retriever=retriever, answerer=fake_answerer)
    thread = {"configurable": {"thread_id": "test-thread"}}
    return graph, retriever, thread


def test_happy_path_resolved():
    graph, _, thread = _build(score=0.6)

    # First invoke runs retrieve -> answer, then PAUSES at the interrupt.
    paused = graph.invoke(initial_state("How do I set env vars?"), thread)
    assert "__interrupt__" in paused
    assert paused["answer"] == "Fake answer to: How do I set env vars?"

    # Resume the same thread with the human's choice; graph runs to END.
    final = graph.invoke(Command(resume="resolved"), thread)
    assert final["resolved"] is True
    assert final["user_action"] == "resolved"
    assert final["ticket_draft"] is None


def test_user_requests_ticket():
    graph, _, thread = _build(score=0.6)
    graph.invoke(initial_state("Why does my build fail?"), thread)

    final = graph.invoke(Command(resume="ticket"), thread)
    assert final["resolved"] is False
    assert final["ticket_draft"] is not None
    assert "Why does my build fail?" in final["ticket_draft"].summary
    # Docs-checked list came from chunk metadata.
    assert final["ticket_draft"].docs_checked == ["Env vars — https://docs.aws/a.html"]


def test_low_confidence_escalates_without_answering():
    graph, _, thread = _build(score=0.2)  # below MIN_TOP_SCORE=0.35

    final = graph.invoke(initial_state("Kubernetes ingress?"), thread)
    assert "__interrupt__" not in final  # never paused: no answer to confirm
    assert final["answer"] is None
    assert final["ticket_draft"] is not None
    assert "No automated answer was produced." in final["ticket_draft"].description


def test_retry_loop_widens_k_and_escalates_at_cap():
    graph, retriever, thread = _build(score=0.6)
    graph.invoke(initial_state("q"), thread)

    # Keep answering "retry" until the cap forces an escalation.
    result = graph.invoke(Command(resume="retry"), thread)
    for _ in range(MAX_ATTEMPTS):
        if "__interrupt__" not in result:
            break
        result = graph.invoke(Command(resume="retry"), thread)

    assert result["ticket_draft"] is not None  # exhausted -> escalated
    assert result["attempts"] == MAX_ATTEMPTS
    # Each retry widened the search instead of repeating it: k = 4 + 2*attempts.
    assert retriever.k_values == [4 + 2 * i for i in range(MAX_ATTEMPTS)]


def test_bad_resume_value_is_rejected():
    graph, _, thread = _build(score=0.6)
    graph.invoke(initial_state("q"), thread)
    with pytest.raises(ValueError, match="Unexpected resume value"):
        graph.invoke(Command(resume="yolo"), thread)
