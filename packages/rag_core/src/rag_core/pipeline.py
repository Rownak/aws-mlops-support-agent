"""
The RagCore facade: wires config, sources, embeddings, vector store and
generation into the ingest verbs (``ingest_documents``, ``ingest_directory``,
``sync``) and the query verbs (``query``, ``aquery``).

All three ingest entry points share one path — ``_prepare_document`` (pure,
never writes) feeding ``_write_prepared`` (the only mutator) — so partial-write
recovery and stale-version replacement behave identically however you ingest.

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
from pathlib import Path
from typing import TypedDict

from langchain_core.documents import Document

from rag_core.config import RagConfig, check_readiness, load_config
from rag_core.embeddings.factory import get_embedding
from rag_core.generation.answer import Answer
from rag_core.generation.generator import AnswerGenerator
from rag_core.llm.factory import get_llm
from rag_core.loaders.markitdown_loader import MarkItDownLoader
from rag_core.processing.chunking import build_chunks, splitter_from_config
from rag_core.processing.hashing import sha256_file_from_path
from rag_core.retriever.confidence import RetrievalConfidence, assess_confidence
from rag_core.retriever.retrieve import retrieve, retrieve_scored
from rag_core.sources import build_sources
from rag_core.vectorstores.pinecone_store import PineconeStore

logger = logging.getLogger(__name__)

#: Chunks per add_documents call. One call for a large document is a single
#: oversized request that can exceed body limits or time out.
DEFAULT_BATCH_SIZE = 100


class IngestError(TypedDict):
    file: str
    error: str


class IngestStats(TypedDict):
    """What one ingestion run did."""

    total: int
    processed: int
    skipped: int
    failed: int
    chunks_created: int
    #: Documents whose content changed since a previous ingest, where the
    #: older version's chunks were removed before writing the new one.
    replaced: int
    errors: list[IngestError]


def _empty_stats(total: int = 0) -> IngestStats:
    return {
        "total": total,
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "chunks_created": 0,
        "replaced": 0,
        "errors": [],
    }


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
        self.batch_size = DEFAULT_BATCH_SIZE

    # ---------------------------------------------------------------- ingest

    def _prepare_document(
        self, file_path: str, splitter, *, extra_metadata: dict | None = None
    ) -> dict:
        """
        Hash, dedup-check, load and split one file. Never writes.

        Read-only with respect to the vector store, so every mutation stays in
        `_write_prepared` and there is exactly one place partial state can be
        created. Failures are captured in the record rather than raised, so one
        bad file cannot abort a whole run — a source's bad `extra_metadata`
        included, which lands here as this file's error rather than killing
        the run.

        Args:
            file_path: Path to the document
            splitter: The splitter to chunk its text with
            extra_metadata: Extra metadata for this file's chunks (see
                `Source.metadata_for`)

        Returns:
            A record whose ``action`` is "skip", "write" or "error".
        """
        record = {
            "file": file_path, "hash": None, "chunks": None, "error": None,
            "action": "error", "stored": 0, "expected": None, "was_partial": False,
        }

        try:
            file_hash = sha256_file_from_path(file_path)
            record["hash"] = file_hash

            # A previous run that died partway leaves chunks behind, and those
            # must be cleared, not mistaken for a finished document.
            status, stored, expected = self.store.get_ingest_status(file_hash)
            record["stored"], record["expected"] = stored, expected

            if status == "complete":
                record["action"] = "skip"
                return record
            record["was_partial"] = status == "partial"

            result = self.loader.load(file_path)
            if not result["success"]:
                record["error"] = result["error"]
                return record

            record["chunks"] = build_chunks(
                text=result["text_content"],
                file_path=file_path,
                file_name=result["file_name"],
                file_type=result["file_type"],
                file_hash=file_hash,
                splitter=splitter,
                extra_metadata=extra_metadata,
            )

            if not record["chunks"]:
                # Scanned/image-only PDFs and empty files land here. Counting
                # them in neither processed nor failed would make them invisible.
                record["error"] = (
                    "no extractable text (possibly a scanned or image-only document)"
                )
                return record

            record["action"] = "write"
            return record

        except Exception as e:
            logger.debug(f"Preparation failed for {file_path}: {e}", exc_info=True)
            record["error"] = str(e)
            return record

    def _write_chunks(self, chunks: list[Document], vectorstore) -> None:
        """Add chunks to the vector store in batches.

        A single call for a large document is one oversized request that can
        exceed body limits or time out.
        """
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        for index in range(total_batches):
            start = index * self.batch_size
            vectorstore.add_documents(chunks[start : start + self.batch_size])

    def _write_prepared(self, record: dict, vectorstore, stats: IngestStats) -> None:
        """
        Write one prepared document and update stats, both in place.

        Sequential by design: it owns every mutation of the collection and of
        the stats dict, so rollback has exactly one place to live.
        """
        file_path = record["file"]
        file_hash = record["hash"]

        if record["action"] == "skip":
            logger.info(f"Skipping (already ingested): {file_path}")
            stats["skipped"] += 1
            return

        if record["action"] == "error":
            stats["failed"] += 1
            stats["errors"].append({"file": file_path, "error": record["error"]})
            logger.error(f"Failed to process {file_path}: {record['error']}")
            return

        chunks = record["chunks"]
        try:
            if record["was_partial"]:
                logger.warning(
                    f"Found incomplete ingest for {file_path} "
                    f"({record['stored']}/{record['expected']} chunks); "
                    "clearing and re-ingesting"
                )
                self.store.delete_by_file_hash(file_hash)

            # The file's content changed, so its hash changed too. Without this
            # the previous version stays in the collection alongside the new
            # one and keeps surfacing in results.
            stale = self.store.delete_by_source(file_path, except_file_hash=file_hash)
            if stale:
                stats["replaced"] += 1
                logger.info(
                    f"Replaced {stale} chunk(s) from a previous version of {file_path}"
                )

            self._write_chunks(chunks, vectorstore)

            stats["chunks_created"] += len(chunks)
            stats["processed"] += 1
            logger.info(f"Processed {file_path}: {len(chunks)} chunks")

        except Exception as e:
            stats["failed"] += 1
            stats["errors"].append({"file": file_path, "error": str(e)})
            logger.error(f"Error writing {file_path}: {e}", exc_info=True)

            # Roll back anything that landed before the failure, so the file is
            # retried cleanly next run instead of being skipped as complete.
            if file_hash:
                try:
                    self.store.delete_by_file_hash(file_hash)
                except Exception as cleanup_error:
                    logger.warning(
                        f"Could not roll back partial ingest for {file_path}: "
                        f"{cleanup_error}"
                    )

    def ingest_documents(
        self,
        file_paths: list[str],
        *,
        extra_metadata: dict[str, dict] | None = None,
    ) -> IngestStats:
        """
        Ingest a specific list of documents into the vector store.

        Files already fully ingested are skipped (matched on content hash, so
        an unchanged file is free to re-run). A file whose content changed
        replaces its older version; a previous run that died partway is
        cleared and re-ingested rather than left truncated.

        Args:
            file_paths: Paths of the documents to ingest
            extra_metadata: Optional per-path extra chunk metadata, keyed by
                the same strings as `file_paths` — typically what each
                `Source.metadata_for()` returned, which is how `sync()` uses
                it. A path with no entry simply gets none. Keys must avoid
                `processing.chunking.RESERVED_METADATA_KEYS`; a collision
                fails that one file, not the run.

        Returns:
            IngestStats — counts plus a per-file error list.

        Example:
            >>> stats = rag.ingest_documents(["a.pdf", "b.md"])  # doctest: +SKIP
            >>> print(f"Processed {stats['processed']} of {stats['total']}")
        """
        stats = _empty_stats(len(file_paths))
        if not file_paths:
            return stats

        use_sparse = self.config.vectorstore.use_sparse
        self.store.create_collection(use_sparse=use_sparse)
        vectorstore = self.store.get_store(use_sparse=use_sparse)
        splitter = splitter_from_config(self.config.splitter)

        logger.info(f"Starting ingestion of {len(file_paths)} document(s)")
        for file_path in file_paths:
            extra = (extra_metadata or {}).get(file_path, {})
            self._write_prepared(
                self._prepare_document(file_path, splitter, extra_metadata=extra),
                vectorstore,
                stats,
            )

        logger.info(
            f"Ingestion complete: {stats['processed']}/{stats['total']} documents "
            f"({stats['skipped']} skipped, {stats['failed']} failed, "
            f"{stats['replaced']} replaced, {stats['chunks_created']} chunks)"
        )
        return stats

    def ingest_directory(
        self,
        directory: str,
        recursive: bool = False,
        extensions: list[str] | None = None,
    ) -> IngestStats:
        """
        Ingest every supported document in a directory.

        Args:
            directory: Directory to scan
            recursive: Whether to descend into subdirectories
            extensions: Extensions to include; defaults to the config's
                ``loader.extensions``

        Returns:
            IngestStats, as `ingest_documents`

        Raises:
            ValueError: If `directory` is not a directory. A missing path is an
                error rather than an empty run, since the two are worth telling
                apart.

        Example:
            >>> stats = rag.ingest_directory("data/", recursive=True)  # doctest: +SKIP
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        exts = [e.lower() for e in (extensions or self.config.loader_extensions)]
        pattern = "**/*" if recursive else "*"
        file_paths = sorted(
            str(p)
            for p in dir_path.glob(pattern)
            if p.is_file() and p.suffix.lower() in exts
        )

        if not file_paths:
            logger.warning(f"No supported files found in {directory} (extensions: {exts})")
            return _empty_stats()

        logger.info(f"Found {len(file_paths)} file(s) in {directory}")
        return self.ingest_documents(file_paths)

    def sync(self) -> IngestStats:
        """
        Reconcile the collection against every configured source.

        Lists what the sources currently hold, then hands those paths to
        `ingest_documents`, so syncing and explicit ingestion share one code
        path (and one set of guarantees around partial writes and replacement).
        Each source is also asked for any extra per-file metadata it wants on
        its chunks (see `Source.metadata_for`).

        Returns:
            IngestStats, as `ingest_documents`
        """
        sources = build_sources([spec.as_dict() for spec in self.config.sources])
        file_paths: list[str] = []
        extra_metadata: dict[str, dict] = {}
        for source in sources:
            paths = source.list_files()
            file_paths.extend(paths)
            # getattr, not a direct call: a source only has to provide
            # list_files() to work here, so duck-typed sources (and tests'
            # fakes) predating metadata_for keep working untouched.
            metadata_for = getattr(source, "metadata_for", None)
            if metadata_for is None:
                continue
            for path in paths:
                metadata = metadata_for(path)
                if metadata:
                    extra_metadata[path] = metadata

        logger.info(f"Sync: {len(file_paths)} file(s) listed by {len(sources)} source(s)")
        return self.ingest_documents(file_paths, extra_metadata=extra_metadata or None)

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
