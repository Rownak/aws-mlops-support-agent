"""
Pinecone vector store wrapper for the RAG pipeline.

Same responsibilities as RagWire's ``QdrantStore``: connect to the backend,
manage the collection (Pinecone calls it an index) lifecycle, and answer the
ingest-reconciliation questions ``pipeline``/sync logic needs — is this file
already fully ingested, and which source paths does the collection currently
hold.

Pinecone's serverless API has no facet/payload-index concept (that part of
``QdrantStore`` — ``create_payload_indexes``, ``get_field_values`` — has no
equivalent here and is intentionally not ported), so this wrapper covers the
subset of the surface that is meaningful for any backend: collection
lifecycle, hybrid dense/sparse retrieval, and hash-based ingest state.
"""

import logging
from typing import Optional, Any, List

logger = logging.getLogger(__name__)

# Pinecone needs the vector dimension at index-creation time; it is a fixed
# property of the embedding model. Unknown model -> fail loudly, don't guess.
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "nomic-embed-text": 768,
    "qwen3-embedding:0.6b": 1024,
}


class PineconeStore:
    """
    Pinecone vector store wrapper with hybrid search support.

    Manages connection to a Pinecone index and provides a high-level
    interface for document storage and retrieval.

    Attributes:
        client: Pinecone client instance
        embedding: Embedding model instance
        index_name: Name of the Pinecone index
        is_local: True when backed by local file storage (Pinecone Local, the
            Docker-based emulator) rather than the managed cloud service.
            Local storage has no serverless/pod spec and no dimension check
            against a live region, so callers should not assume production
            guarantees against it.

    Example:
        >>> store = PineconeStore(config, embedding)
        >>> store.set_collection("financial_docs")
        >>> vectorstore = store.get_store()
        >>> docs = vectorstore.similarity_search("query", k=5)
    """

    #: Overridden per-instance in __init__. Declared here so instances built by
    #: other means (tests, subclasses) still have a sane default.
    is_local = False

    def __init__(
        self, config: dict, embedding: Any, collection_name: Optional[str] = None
    ):
        """
        Initialize the Pinecone vector store.

        Args:
            config: Configuration dictionary with Pinecone settings
            embedding: Embedding model instance
            collection_name: Optional index name to use

        Raises:
            ImportError: If the pinecone package is not installed
        """
        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError(
                "pinecone is required. Install with: pip install pinecone"
            )

        api_key = config.get("api_key")
        host = config.get("host")  # set for Pinecone Local / a self-hosted proxy

        if host:
            # Pinecone Local (the Docker emulator) or a self-hosted proxy talks
            # over plain HTTP with no real API key required.
            self.client = Pinecone(api_key=api_key or "local", host=host)
            self.is_local = True
            logger.info(f"Using local Pinecone storage at {host}")
            logger.info(
                "Local Pinecone storage does not enforce serverless region/cloud "
                "settings and is meant for development. Run against the managed "
                "service for production workloads."
            )
        else:
            if not api_key:
                raise ValueError(
                    "vectorstore.api_key (or PINECONE_API_KEY) is required to "
                    "connect to the managed Pinecone service."
                )
            self.client = Pinecone(api_key=api_key)
            self.is_local = False
            logger.info("Connected to Pinecone (managed service)")

        self.embedding = embedding
        self.collection_name = collection_name
        self.config = config

    def _index(self, name: Optional[str] = None):
        """Get an Index handle, forcing http:// against Pinecone Local.

        `describe_index` returns a scheme-less host (e.g. "localhost:5081"),
        and the SDK defaults a scheme-less host to https:// — which Pinecone
        Local doesn't speak, causing an SSL error. Managed Pinecone hosts
        already come back with https:// and are unaffected.
        """
        name = name or self.collection_name
        if not self.is_local:
            return self.client.Index(name)

        host = self.client.describe_index(name).host
        if not host.startswith(("http://", "https://")):
            host = f"http://{host}"
        return self.client.Index(host=host)

    def set_collection(self, name: str) -> None:
        """
        Set the index name for operations.

        Args:
            name: Index name to use
        """
        self.collection_name = name
        logger.info(f"Index set to: {name}")

    def get_store(self, use_sparse: bool = False) -> Any:
        """
        Get the LangChain PineconeVectorStore instance.

        Args:
            use_sparse: Whether to enable hybrid search with sparse vectors.
                Pinecone hybrid search needs a dotproduct-metric index and a
                sparse encoder; unlike Qdrant's single-collection hybrid mode,
                sparse values are supplied per-upsert rather than declared on
                the vector store object, so this flag currently only guards
                against use on a non-dotproduct index.

        Returns:
            PineconeVectorStore instance configured with current settings

        Raises:
            ValueError: If collection_name is not set, or hybrid is requested
                against an index whose metric cannot support it
        """
        if not self.collection_name:
            raise ValueError("Index name not set. Call set_collection() first.")

        try:
            from langchain_pinecone import PineconeVectorStore
        except ImportError:
            raise ImportError(
                "langchain-pinecone is required. Install with: pip install langchain-pinecone"
            )

        if use_sparse:
            info = self.get_collection_info()
            metric = getattr(getattr(info, "spec", None), "metric", None) or info.get("metric") \
                if isinstance(info, dict) else getattr(info, "metric", None)
            if metric and metric != "dotproduct":
                raise ValueError(
                    f"Index '{self.collection_name}' uses metric '{metric}'; hybrid "
                    "(dense + sparse) search requires a 'dotproduct' index."
                )

        return PineconeVectorStore(
            index=self._index(),
            embedding=self.embedding,
        )

    def create_collection(
        self, collection_name: Optional[str] = None, use_sparse: bool = True
    ) -> None:
        """
        Create a new Pinecone index if it does not already exist.

        Args:
            collection_name: Name of index (uses current if not provided)
            use_sparse: Whether to provision the index with the dotproduct
                metric hybrid search needs. Dense-only indexes use cosine.
        """
        name = collection_name or self.collection_name
        if not name:
            raise ValueError("Index name must be provided")

        if self.client.has_index(name):
            logger.info(f"Index '{name}' already exists, skipping creation")
            return

        test_embedding = self.embedding.embed_query("test")
        vector_size = len(test_embedding)
        metric = "dotproduct" if use_sparse else "cosine"

        # The SDK requires `spec` even against Pinecone Local, which ignores
        # its cloud/region values but still validates the object shape.
        from pinecone import ServerlessSpec

        cloud = self.config.get("cloud", "aws")
        region = self.config.get("region", "us-east-1")
        self.client.create_index(
            name=name,
            dimension=vector_size,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

        logger.info(f"Created index '{name}' (metric={metric}, dim={vector_size})")

    def delete_collection(self, collection_name: Optional[str] = None) -> None:
        """
        Delete a Pinecone index.

        Args:
            collection_name: Name of index to delete
        """
        name = collection_name or self.collection_name
        if not name:
            raise ValueError("Index name must be provided")

        self.client.delete_index(name)
        logger.info(f"Deleted index: {name}")

    def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """
        Check if an index exists.

        Args:
            collection_name: Name of index to check

        Returns:
            True if the index exists, False otherwise
        """
        name = collection_name or self.collection_name
        if not name:
            return False
        return self.client.has_index(name)

    def _file_hash_filter(self, file_hash: str) -> dict:
        """Build a Pinecone metadata filter matching every chunk of one source file."""
        return {"file_hash": {"$eq": file_hash}}

    def file_hash_exists(self, file_hash: str) -> bool:
        """
        Check whether any chunk of a file is present, by its SHA256 hash.

        Note: presence is not the same as a complete ingest, because a run that
        failed partway leaves chunks behind. Use get_ingest_status() to
        distinguish.

        Args:
            file_hash: SHA256 hash of the file content

        Returns:
            True if at least one chunk with this file_hash exists in the index
        """
        return self.count_by_file_hash(file_hash) > 0

    def count_by_file_hash(self, file_hash: str) -> int:
        """
        Count how many chunks of a given file are stored in the index.

        Pinecone has no exact count-by-filter API, so this queries the nearest
        neighbours of a zero vector under the filter and reports how many
        matched, capped at `top_k`. That is exact up to the cap and is only
        used to detect presence/partial-vs-complete, not for reporting exact
        totals on very large files.

        Args:
            file_hash: SHA256 hash of the file content

        Returns:
            Number of stored chunks found (0 if the index or file is absent)
        """
        if not self.collection_name or not self.collection_exists():
            return 0

        index = self._index()
        dimension = self.get_vector_size() or len(self.embedding.embed_query("test"))
        result = index.query(
            vector=[0.0] * dimension,
            filter=self._file_hash_filter(file_hash),
            top_k=10000,
            include_metadata=False,
        )
        return len(result.matches)

    def get_ingest_status(self, file_hash: str) -> tuple:
        """
        Determine whether a file is fully ingested, partially ingested, or absent.

        Every chunk records the document's ``total_chunks``, so a complete
        ingest is one where the stored chunk count matches that number. A run
        that died partway through upserting leaves fewer. Without this check
        the leftover chunks make the file look already-ingested and it is
        skipped forever, leaving the document permanently truncated.

        Args:
            file_hash: SHA256 hash of the file content

        Returns:
            Tuple of (status, stored_count, expected_count) where status is
            "absent", "partial", or "complete". expected_count is None when
            nothing is stored.
        """
        if not self.collection_name or not self.collection_exists():
            return ("absent", 0, None)

        index = self._index()
        dimension = self.get_vector_size() or len(self.embedding.embed_query("test"))
        result = index.query(
            vector=[0.0] * dimension,
            filter=self._file_hash_filter(file_hash),
            top_k=1,
            include_metadata=True,
        )
        stored = self.count_by_file_hash(file_hash)
        if stored == 0:
            return ("absent", 0, None)

        expected = None
        if result.matches:
            expected = (result.matches[0].metadata or {}).get("total_chunks")

        if not isinstance(expected, (int, float)) or expected <= 0:
            # No usable marker, so treat presence as complete rather than
            # re-ingesting data written by an older version of the engine.
            return ("complete", stored, None)

        expected = int(expected)
        return ("complete" if stored >= expected else "partial", stored, expected)

    def delete_by_file_hash(self, file_hash: str) -> int:
        """
        Delete every stored chunk belonging to one source file.

        Used to clear a partial ingest before retrying, and to remove a
        document that is being replaced.

        Args:
            file_hash: SHA256 hash of the file content

        Returns:
            Number of chunks that were present before deletion
        """
        count = self.count_by_file_hash(file_hash)
        if count == 0:
            return 0

        index = self._index()
        index.delete(filter=self._file_hash_filter(file_hash))
        logger.info(f"Deleted {count} chunk(s) for file_hash {file_hash[:12]}…")
        return count

    def delete_by_source(
        self, source: str, except_file_hash: Optional[str] = None
    ) -> int:
        """
        Delete stored chunks by their source path, optionally sparing one version.

        Deduplication is keyed on file content, so an edited document hashes
        differently and would otherwise be stored *alongside* its previous
        version, leaving the old text retrievable forever. Passing the new
        hash as ``except_file_hash`` removes only the stale copies.

        Args:
            source: Source path recorded in chunk metadata
            except_file_hash: File hash to preserve (the version being written)

        Returns:
            Number of chunks deleted
        """
        if not self.collection_name or not self.collection_exists():
            return 0

        index = self._index()
        dimension = self.get_vector_size() or len(self.embedding.embed_query("test"))
        filter_ = {"source": {"$eq": source}}
        result = index.query(
            vector=[0.0] * dimension,
            filter=filter_,
            top_k=10000,
            include_metadata=True,
        )
        ids = [
            match.id
            for match in result.matches
            if not except_file_hash
            or (match.metadata or {}).get("file_hash") != except_file_hash
        ]
        if not ids:
            return 0

        index.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} stale chunk(s) for source: {source}")
        return len(ids)

    def list_sources(self) -> List[str]:
        """
        Return every distinct source path stored in the index.

        Pinecone has no collection-wide scroll/list API, so this pages through
        `list()` (which returns vector IDs, not metadata) and fetches metadata
        in batches. Reconciliation needs the complete set: a source missing
        from a truncated listing looks like a source that was never ingested,
        and would be re-ingested on every sync.

        Returns:
            Distinct ``source`` metadata values, in first-seen order
        """
        if not self.collection_name or not self.collection_exists():
            return []

        index = self._index()
        sources: List[str] = []
        seen = set()

        for id_batch in index.list():
            if not id_batch:
                continue
            fetched = index.fetch(ids=list(id_batch))
            for vector in fetched.vectors.values():
                source = (vector.metadata or {}).get("source")
                if source and source not in seen:
                    seen.add(source)
                    sources.append(source)

        return sources

    def get_collection_info(self, collection_name: Optional[str] = None) -> dict:
        """
        Get information about an index.

        Args:
            collection_name: Name of index

        Returns:
            Pinecone's index description
        """
        name = collection_name or self.collection_name
        if not name:
            raise ValueError("Index name must be provided")
        return self.client.describe_index(name)

    def get_vector_size(self, collection_name: Optional[str] = None) -> Optional[int]:
        """
        Return the dense vector dimension an existing index was created with.

        Args:
            collection_name: Name of index (uses current if not provided)

        Returns:
            Vector dimension, or None if it cannot be determined
        """
        try:
            info = self.get_collection_info(collection_name)
            return info.dimension
        except Exception as e:
            logger.debug(f"Could not read vector size: {e}")
            return None
