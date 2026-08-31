"""Task 3.4 — Wire the nodes into a graph with conditional edges.

Shape (linear from task 3.2, plus the routing added in 3.3/3.4):

    START → retrieve ──confident──→ answer → confirm_resolution ──resolved──→ END
                │                                   │        │
                └──low confidence──→ escalate ←──ticket    retry
                                        │                    │ (attempts left?)
                                       END ←─exhausted───────┴──→ retrieve

A conditional edge is just a function `state -> next node name` registered
with `add_conditional_edges`. The two routers below are plain pure
functions so they're unit-testable without building a graph at all.

Escalation triggers (task 3.4), each visible in exactly one place:
  1. low retrieval confidence      -> route_after_retrieve
  2. user asks for a ticket        -> route_after_confirm ("ticket")
  3. retry attempts exhausted      -> route_after_confirm ("retry" at the cap)
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from rag_core.config import (
    EmbeddingConfig,
    GenerationConfig,
    LLMConfig,
    RagConfig,
    RetrieverConfig,
    SplitterConfig,
    VectorStoreConfig,
)

from aws_mlops_support_agent.agent.nodes import build_nodes
from aws_mlops_support_agent.agent.state import AgentState
from aws_mlops_support_agent.settings import AgentConfig

# Max retrieve→answer cycles before a "retry" turns into an escalation.
# 2 keeps the demo snappy; the counter lives in state (see state.py).
MAX_ATTEMPTS = 2

# The checkpointer serializes state with msgpack; custom classes must be
# allowlisted or deserialization warns now and will be BLOCKED in a future
# langgraph release (arbitrary-class deserialization is a code-exec risk if
# an attacker can write to the checkpoint store). New state types go here.
# langchain_core.documents.Document (in `chunks`) needs no entry: LangGraph's
# serializer revives LangChain "core" Serializable objects separately from
# this msgpack allowlist.
STATE_TYPES_ALLOWLIST = [
    ("rag_core.retriever.confidence", "RetrievalConfidence"),
    ("aws_mlops_support_agent.agent.ticket", "TicketDraft"),
]


# Task 5.2 — LangGraph auto-traces NODES when LANGSMITH_TRACING is on, but
# conditional-edge functions run inside the graph machinery without their own
# span; @traceable gives each routing decision one (output = chosen node).
@traceable
def route_after_retrieve(state: AgentState) -> str:
    """Weak retrieval -> don't bluff an answer, go straight to escalation."""
    if state["confidence"] is not None and state["confidence"].is_confident:
        return "answer"
    return "escalate"


@traceable
def route_after_confirm(state: AgentState) -> str:
    """Turn the human's interrupt choice into the next hop."""
    action = state["user_action"]
    if action == "resolved":
        return END
    if action == "ticket":
        return "escalate"
    # "retry": loop back only while attempts remain, else escalate.
    return "retrieve" if state["attempts"] < MAX_ATTEMPTS else "escalate"


def build_graph(cfg: AgentConfig, retriever=None, answerer=None, jira_creator=None):
    """Compile the agent graph; pass fakes for retriever/answerer/jira_creator in tests.

    The checkpointer is REQUIRED for the interrupt in confirm_resolution:
    it's what saves state when the graph pauses and reloads it on
    Command(resume=...). InMemorySaver keeps checkpoints in this process —
    fine for a CLI session; a server deployment would swap in a persistent
    saver (SQLite/Postgres) without touching the graph.
    """
    nodes = build_nodes(cfg, retriever=retriever, answerer=answerer, jira_creator=jira_creator)

    builder = StateGraph(AgentState)
    for name, fn in nodes.items():
        builder.add_node(name, fn)

    builder.add_edge(START, "retrieve")
    builder.add_conditional_edges("retrieve", route_after_retrieve, ["answer", "escalate"])
    builder.add_edge("answer", "confirm_resolution")
    builder.add_conditional_edges(
        "confirm_resolution", route_after_confirm, ["retrieve", "escalate", END]
    )
    builder.add_edge("escalate", END)

    serde = JsonPlusSerializer(allowed_msgpack_modules=STATE_TYPES_ALLOWLIST)
    return builder.compile(checkpointer=InMemorySaver(serde=serde))


if __name__ == "__main__":
    # Visualize the graph:  uv run python -m aws_mlops_support_agent.agent.graph
    #
    # Prints a Mermaid diagram of the compiled graph — paste it into
    # https://mermaid.live, a GitHub markdown block, or the README to see
    # nodes and conditional edges rendered. Kept out of build_graph() so
    # tests and the app don't dump it on every run.
    #
    # No-op fakes are injected so this needs NO API keys and never opens a
    # Pinecone/OpenAI connection — the placeholder Config is never read.
    # embeddings/llm default to provider="ollama" (needs no key); vectorstore
    # must set `host` or RagConfig._validate() demands PINECONE_API_KEY even
    # though nothing here ever calls Pinecone.
    stub_cfg = AgentConfig(
        rag=RagConfig(
            embeddings=EmbeddingConfig(),
            llm=LLMConfig(),
            vectorstore=VectorStoreConfig(host="http://localhost:5080"),
            splitter=SplitterConfig(),
            retriever=RetrieverConfig(),
            generation=GenerationConfig(),
            sources=(),
            loader_extensions=(),
        ),
        jira_base_url=None,
        jira_email=None,
        jira_api_token=None,
        jira_project_key=None,
        dry_run=True,
    )
    graph = build_graph(stub_cfg, retriever=lambda q, k=4: [], answerer=lambda q, c: "")
    drawable = graph.get_graph()

    print("Mermaid diagram (paste into https://mermaid.live or a ```mermaid block):\n")
    print(drawable.draw_mermaid())

    # ASCII rendering needs the optional `grandalf` package; skip gracefully.
    try:
        drawable.print_ascii()
    except ImportError:
        print("(For an ASCII rendering in the terminal: uv add --dev grandalf)")
