"""Task 1.4 — Create the Pinecone serverless index, embed chunks, upsert.

The embedding model name comes from Config — the same value the Phase 2
retriever will read — so ingestion and query can never drift apart.
"""

import time

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import Config

# Pinecone needs the vector dimension at index-creation time; it is a fixed
# property of the embedding model. Unknown model -> fail loudly, don't guess.
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def get_vector_store(cfg: Config) -> PineconeVectorStore:
    """Ensure the index exists (right dimension) and return a LangChain wrapper.

    Also used by the sanity-check CLI and, later, the Phase 2 retriever.
    """
    dimension = EMBEDDING_DIMENSIONS.get(cfg.openai_embedding_model)
    if dimension is None:
        raise RuntimeError(
            f"Unknown embedding model '{cfg.openai_embedding_model}': add its "
            "dimension to EMBEDDING_DIMENSIONS in src/ingest/index.py"
        )

    pc = Pinecone(api_key=cfg.pinecone_api_key)

    if not pc.has_index(cfg.pinecone_index_name):
        print(f"[index] creating serverless index '{cfg.pinecone_index_name}' (dim={dimension})")
        pc.create_index(
            name=cfg.pinecone_index_name,
            dimension=dimension,
            metric="cosine",  # standard choice for OpenAI embeddings
            spec=ServerlessSpec(cloud="aws", region=cfg.aws_region),
        )
        # Index creation is async; wait until it can accept upserts.
        while not pc.describe_index(cfg.pinecone_index_name).status["ready"]:
            time.sleep(1)

    existing = pc.describe_index(cfg.pinecone_index_name)
    if existing.dimension != dimension:
        raise RuntimeError(
            f"Index '{cfg.pinecone_index_name}' has dimension {existing.dimension} but "
            f"model '{cfg.openai_embedding_model}' produces {dimension}. Delete the "
            "index or change PINECONE_INDEX_NAME / OPENAI_EMBEDDING_MODEL."
        )

    embeddings = OpenAIEmbeddings(model=cfg.openai_embedding_model, api_key=cfg.openai_api_key)
    return PineconeVectorStore(index=pc.Index(cfg.pinecone_index_name), embedding=embeddings)


def upsert_chunks(cfg: Config, docs: list[Document]) -> None:
    """Embed all chunks and upsert them under their deterministic IDs.

    Caveat: if a source doc SHRINKS between runs, its highest-index chunks
    from the previous run are not overwritten and linger as orphans. Fine
    for this frozen (~2023) corpus; revisit if the corpus becomes live.
    """
    store = get_vector_store(cfg)
    ids = [doc.id for doc in docs]
    print(f"[index] embedding + upserting {len(docs)} chunks (model={cfg.openai_embedding_model})")
    # add_documents batches the OpenAI embedding calls and Pinecone upserts.
    store.add_documents(docs, ids=ids)
    print("[index] upsert complete")
