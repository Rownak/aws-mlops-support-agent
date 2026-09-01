"""
The RagCore facade: wires config, sources, embeddings, vector store and
generation into the ingest verbs (``ingest_documents``, ``ingest_directory``,
``sync``) and the query verbs (``query``, ``aquery``).

The ingest verbs are thin delegations to `ingestion.Ingestor`, which owns
every write to the store — chunking, deduplication, partial-write recovery
and replacement of stale versions. They stay on this class so
``rag.sync()`` remains the obvious entry point; `rag.ingestor` is there for
callers who want the ingest side on its own.

Kept intentionally thin — every module it composes (``sources``, ``processing``,
``embeddings``, ``vectorstores``, ``generation``, ``retriever``) also works
standalone at the function level: ``processing.chunking.build_chunks``,
``retriever.retrieve``,
``retriever.confidence.assess_confidence``, and
``generation.generator.AnswerGenerator.generate``/``agenerate`` are each
independently callable, so an agent can run retrieval and confidence scoring
as its own step — e.g. to escalate instead of generating an answer when the
match is weak — without going through this facade at all. ``query()``/
``aquery()`` below are exactly that composition, kept here as the convenient
default path.
"""

import logging

from langchain_core.documents import Document

from rag_core.config import RagConfig, check_readiness, load_config
from rag_core.embeddings.factory import get_embedding
from rag_core.generation.answer import Answer
from rag_core.generation.generator import AnswerGenerator
from rag_core.ingestion import DEFAULT_BATCH_SIZE, IngestStats, Ingestor
from rag_core.llm.factory import get_llm
from rag_core.loaders.markitdown_loader import MarkItDownLoader
from rag_core.retriever.confidence import RetrievalConfidence, assess_confidence
from rag_core.retriever.retrieve import retrieve, retrieve_scored
from rag_core.vectorstores.pinecone_store import PineconeStore

logger = logging.getLogger(__name__)


class RagCore:
    """
    Corpus-agnostic RAG engine, configured entirely from one YAML file.

    Example:
        >>> rag = RagCore("config.yaml")   # doctest: +SKIP
        >>> rag.sync()                      # doctest: +SKIP
        >>> answer = rag.query("How do I cache dependencies?")  # doctest: +SKIP
        >>> print(answer.formatted())       # doctest: +SKIP
    """

    def __init__(self, config_path: str):
        self.config: RagConfig = load_config(config_path)
        check_readiness(self.config)

        self.embedding = get_embedding(self.config.embeddings.as_dict())
        self.store = PineconeStore(
            self.config.vectorstore.as_dict(),
            self.embedding,
            collection_name=self.config.vectorstore.collection_name,
        )
        self.loader = MarkItDownLoader()
        self.llm = get_llm(self.config.llm.as_dict())
        self.generator = AnswerGenerator(
            llm=self.llm,
            max_context_chars=self.config.generation.max_context_chars,
            system_prompt=self.config.generation.system_prompt,
        )
        self.ingestor = Ingestor(
            self.config, self.store, self.loader, batch_size=DEFAULT_BATCH_SIZE
        )

    # ---------------------------------------------------------------- ingest
    #
    # Thin delegations to `self.ingestor`, which owns every write to the
    # store. They stay here so `rag.sync()` remains the obvious entry point;
    # see `rag_core.ingestion.ingestor` for what actually happens.

    @property
    def batch_size(self) -> int:
        """Chunks per `add_documents` call, as used by the ingestor."""
        return self.ingestor.batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self.ingestor.batch_size = value

    def ingest_documents(
        self,
        file_paths: list[str],
        *,
        extra_metadata: dict[str, dict] | None = None,
    ) -> IngestStats:
        """
        Ingest a specific list of documents into the vector store.

        See :meth:`rag_core.ingestion.Ingestor.ingest_documents`.

        Example:
            >>> stats = rag.ingest_documents(["a.pdf", "b.md"])  # doctest: +SKIP
            >>> print(f"Processed {stats['processed']} of {stats['total']}")
        """
        return self.ingestor.ingest_documents(file_paths, extra_metadata=extra_metadata)

    def ingest_directory(
        self,
        directory: str,
        recursive: bool = False,
        extensions: list[str] | None = None,
    ) -> IngestStats:
        """
        Ingest every supported document in a directory.

        See :meth:`rag_core.ingestion.Ingestor.ingest_directory`.

        Example:
            >>> stats = rag.ingest_directory("data/", recursive=True)  # doctest: +SKIP
        """
        return self.ingestor.ingest_directory(directory, recursive, extensions)

    def sync(self) -> IngestStats:
        """
        Reconcile the collection against every configured source.

        See :meth:`rag_core.ingestion.Ingestor.sync`.
        """
        return self.ingestor.sync()

    def _vectorstore(self):
        return self.store.get_store(use_sparse=self.config.vectorstore.use_sparse)

    def retrieve(
        self, question: str, k: int | None = None, filters: dict | None = None
    ) -> list[Document]:
        """
        Retrieve chunks for a question, best first.

        The rawest view of the query path: no confidence verdict, no LLM call.
        Use it to see what retrieval actually returns — which chunks, in what
        order — when tuning `top_k`, chunking, or a reranker. Each document
        carries its score in `metadata["score"]` when the search type produced
        one; call `retrieve_scored()` when you need the score as a value.

        Args:
            question: The question to retrieve chunks for
            k: Override the configured `retriever.top_k`
            filters: Backend-native metadata filter, passed through untouched

        Returns:
            Documents, best first.

        Example:
            >>> for doc in rag.retrieve("how do I cache?"):  # doctest: +SKIP
            ...     print(doc.metadata["source"], doc.page_content[:120])
        """
        return retrieve(
            question, self._vectorstore(), self.config.retriever, top_k=k, filters=filters
        )

    def retrieve_scored(
        self, question: str, k: int | None = None, filters: dict | None = None
    ) -> list[tuple[Document, float]]:
        """
        Retrieve chunks with their normalized [0, 1] relevance scores.

        For the callers that need the number itself — confidence scoring,
        eval tables, threshold tuning. See
        :func:`rag_core.retriever.retrieve.retrieve_scored` for the reranker
        caveat and the MMR restriction.

        Args:
            question: The question to retrieve chunks for
            k: Override the configured `retriever.top_k`
            filters: Backend-native metadata filter, passed through untouched

        Returns:
            (document, score) pairs, best first.
        """
        return retrieve_scored(
            question, self._vectorstore(), self.config.retriever, top_k=k, filters=filters
        )

    def retrieve_with_confidence(
        self, question: str, k: int | None = None
    ) -> tuple[list, RetrievalConfidence]:
        """
        Retrieve chunks and score how confident the match is, as one step.

        This is the seam an agent calls directly to decide whether to
        generate an answer at all — e.g. escalate to a human/ticket instead
        of calling `generate()` when `confidence.is_confident` is False —
        without going through `query()` or duplicating its retrieval logic.

        Args:
            question: The user's question
            k: Override the configured `retriever.top_k`

        Returns:
            (documents, confidence): plain documents (best first, ready to
            pass to `AnswerGenerator.generate`) and the confidence verdict.
            Call `retrieve_scored()` instead when you want the scores too.
        """
        scored = self.retrieve_scored(question, k=k)
        confidence = assess_confidence(scored, min_top_score=self.config.retriever.min_top_score)
        documents = [doc for doc, _ in scored]
        return documents, confidence

    def query(self, question: str, k: int | None = None) -> Answer:
        """
        Retrieve, score confidence, and generate a grounded, cited Answer.

        Always generates an answer regardless of confidence — the system
        prompt already makes the model admit gaps, so seeing that output is
        useful for tuning. An agent that wants to escalate on low confidence
        instead should call `retrieve_with_confidence()` directly and decide
        for itself whether to reach `generate()` at all.

        Args:
            question: The user's question
            k: Override the configured `retriever.top_k`

        Returns:
            An Answer (see `generation.answer.Answer`)
        """
        documents, confidence = self.retrieve_with_confidence(question, k=k)
        logger.info(f"Retrieval confidence: {confidence.reason}")
        return self.generator.generate(question, documents)

    async def aquery(self, question: str, k: int | None = None) -> Answer:
        """Async version of :meth:`query`. Only generation is awaited."""
        documents, confidence = self.retrieve_with_confidence(question, k=k)
        logger.info(f"Retrieval confidence: {confidence.reason}")
        return await self.generator.agenerate(question, documents)
