"""Create the Pinecone serverless index, embed chunks, upsert.

The embedding model name comes from RagConfig — the same value the retriever
reads — so ingestion and query can never drift apart. That single shared
setting is why this module takes a config rather than model arguments.

Two entry points on purpose:
  - `get_vector_store` may CREATE the index (ingestion is the step that is
    supposed to set the corpus up);
  - `get_vector_store_for_query` must NOT — auto-creating at query time would
    silently paper over a wrong index name with a brand-new empty index.
"""

import time

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from rag_core.config import RagConfig

# Pinecone needs the vector dimension at index-creation time; it is a fixed
# property of the embedding model. Unknown model -> fail loudly, don't guess.
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def embedding_dimension(cfg: RagConfig) -> int:
    dimension = EMBEDDING_DIMENSIONS.get(cfg.openai_embedding_model)
    if dimension is None:
        raise RuntimeError(
            f"Unknown embedding model '{cfg.openai_embedding_model}': add its dimension "
            "to EMBEDDING_DIMENSIONS in rag_core/vectorstore/pinecone_store.py"
        )
    return dimension


def _wrap_store(cfg: RagConfig, pc: Pinecone) -> PineconeVectorStore:
    embeddings = OpenAIEmbeddings(model=cfg.openai_embedding_model, api_key=cfg.openai_api_key)
    return PineconeVectorStore(index=pc.Index(cfg.pinecone_index_name), embedding=embeddings)


def _check_dimension(cfg: RagConfig, pc: Pinecone, dimension: int) -> None:
    """Refuse to use an index whose dimension can't hold this model's vectors."""
    existing = pc.describe_index(cfg.pinecone_index_name)
    if existing.dimension != dimension:
        raise RuntimeError(
            f"Index '{cfg.pinecone_index_name}' has dimension {existing.dimension} but "
            f"model '{cfg.openai_embedding_model}' produces {dimension}. Delete the "
            "index or change PINECONE_INDEX_NAME / OPENAI_EMBEDDING_MODEL."
        )


def get_vector_store(cfg: RagConfig, client=None) -> PineconeVectorStore:
    """Ingestion-time store: ensure the index exists with the right dimension.

    `client` is injectable so tests can exercise the guards with a fake
    Pinecone client and never open a real connection.
    """
    dimension = embedding_dimension(cfg)
    pc = client or Pinecone(api_key=cfg.pinecone_api_key)

    if not pc.has_index(cfg.pinecone_index_name):
        print(f"[index] creating serverless index '{cfg.pinecone_index_name}' (dim={dimension})")
        pc.create_index(
            name=cfg.pinecone_index_name,
            dimension=dimension,
            metric=cfg.pinecone_index_metric,
            spec=ServerlessSpec(cloud="aws", region=cfg.aws_region),
        )
        # Index creation is async; wait until it can accept upserts.
        while not pc.describe_index(cfg.pinecone_index_name).status["ready"]:
            time.sleep(1)

    _check_dimension(cfg, pc, dimension)
    return _wrap_store(cfg, pc)


def get_vector_store_for_query(cfg: RagConfig, client=None) -> PineconeVectorStore:
    """Query-time store: fail loudly if the index doesn't exist, never create it.

    A missing index at query time means misconfiguration (wrong index name, or
    ingestion never ran) — surfacing that as a startup error is much better
    than silently querying a fresh empty index.
    """
    dimension = embedding_dimension(cfg)
    pc = client or Pinecone(api_key=cfg.pinecone_api_key)

    if not pc.has_index(cfg.pinecone_index_name):
        raise RuntimeError(
            f"Pinecone index '{cfg.pinecone_index_name}' does not exist. Check "
            "PINECONE_INDEX_NAME, or run ingestion first."
        )

    _check_dimension(cfg, pc, dimension)
    return _wrap_store(cfg, pc)


def upsert_chunks(cfg: RagConfig, docs: list[Document], store=None) -> None:
    """Embed all chunks and upsert them under their deterministic IDs.

    Caveat: if a source doc SHRINKS between runs, its highest-index chunks
    from the previous run are not overwritten and linger as orphans. Fine for
    a frozen corpus; revisit if the corpus becomes live.
    """
    store = store or get_vector_store(cfg)
    ids = [doc.id for doc in docs]
    print(f"[index] embedding + upserting {len(docs)} chunks (model={cfg.openai_embedding_model})")
    # add_documents batches the OpenAI embedding calls and Pinecone upserts.
    store.add_documents(docs, ids=ids)
    print("[index] upsert complete")
