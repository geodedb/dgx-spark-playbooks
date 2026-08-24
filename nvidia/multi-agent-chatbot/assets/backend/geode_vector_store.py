#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Geode-based vector store for document embedding and retrieval."""

import glob
import json
import os
import uuid
from typing import List, Optional, Callable

import requests
from geode_client import Client, QueryError, TransportType
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

from logger import logger


class CustomEmbeddings:
    """Wraps qwen3 embedding model to match OpenAI format"""
    def __init__(self, model: str = "Qwen3-Embedding-4B-Q8_0.gguf", host: str = "http://qwen3-embedding:8000"):
        self.model = model
        self.url = f"{host}/v1/embeddings"

    def __call__(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            response = requests.post(
                self.url,
                json={"input": text, "model": self.model},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()
            embeddings.append(data["data"][0]["embedding"])
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts."""
        return self.__call__(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        return self.__call__([text])[0]


def _vec_param(embedding: List[float]) -> str:
    """Serialize an embedding for param-bound VectorF32 storage.

    Embeddings must be stored as true VectorF32 values — knn()/HNSW ignore
    LIST-typed properties entirely. The only fully param-bound form geode
    v0.8.5 accepts is `vectorf32(toString($vec))` with $vec bound to the
    JSON string of the float list.
    """
    return json.dumps(embedding)


def _val(row: dict, key: str, default=None):
    """Extract a plain Python value from a Geode row entry."""
    v = row.get(key)
    if v is None:
        return default
    if isinstance(v, (str, int, float, bool)):
        return v
    if hasattr(v, 'as_string'):
        return v.as_string
    return default


class GeodeVectorStore:
    """Vector store for document embedding and retrieval using Geode.

    Decoupled from ConfigManager - uses optional callbacks for source management.
    """

    def __init__(
        self,
        embeddings=None,
        host: str = "geode",
        port: int = 3141,
        username: str = "",
        password: str = "",
        on_source_deleted: Optional[Callable[[str], None]] = None
    ):
        """Initialize the vector store.

        Args:
            embeddings: Embedding model to use (defaults to CustomEmbeddings)
            host: Geode server host
            port: Geode server port
            username: Geode authentication username
            password: Geode authentication password
            on_source_deleted: Optional callback when a source is deleted
        """
        try:
            self.embeddings = embeddings or CustomEmbeddings(model="qwen3-embedding-custom")
            self.host = host
            self.port = port
            self.username = username
            self.password = password
            self.on_source_deleted = on_source_deleted
            transport_str = os.getenv("GEODE_TRANSPORT", "quic").lower()
            transport = TransportType.GRPC if transport_str == "grpc" else TransportType.QUIC
            kwargs = {"skip_verify": True, "transport_type": transport}
            if username:
                kwargs["username"] = username
            if password:
                kwargs["password"] = password
            self.client = Client(host=self.host, port=self.port, **kwargs)
            self._index_ensured = False

            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            logger.debug({
                "message": "GeodeVectorStore initialized successfully"
            })
        except Exception as e:
            logger.error({
                "message": "Error initializing GeodeVectorStore",
                "error": str(e)
            }, exc_info=True)
            raise

    def _initialize_store(self):
        """No-op for API compatibility. Geode does not require explicit store initialization."""
        logger.debug({
            "message": "Geode vector store ready",
            "host": self.host,
            "port": self.port
        })

    def _load_documents(self, file_paths: List[str] = None, input_dir: str = None) -> List[Document]:
        try:
            documents = []
            source_name = None

            if input_dir:
                source_name = os.path.basename(os.path.normpath(input_dir))
                logger.debug({
                    "message": "Loading files from directory",
                    "directory": input_dir,
                    "source": source_name
                })
                file_paths = glob.glob(os.path.join(input_dir, "**"), recursive=True)
                file_paths = [f for f in file_paths if os.path.isfile(f)]

            logger.info(f"Processing {len(file_paths)} files: {file_paths}")

            for file_path in file_paths:
                try:
                    if not source_name:
                        source_name = os.path.basename(file_path)
                        logger.info(f"Using filename as source: {source_name}")

                    logger.info(f"Loading file: {file_path}")

                    file_ext = os.path.splitext(file_path)[1].lower()
                    logger.info(f"File extension: {file_ext}")

                    try:
                        loader = UnstructuredLoader(file_path)
                        docs = loader.load()
                        logger.info(f"Successfully loaded {len(docs)} documents from {file_path}")
                    except Exception as pdf_error:
                        logger.error('error with unstructured loader, trying to load from scratch')
                        file_text = None
                        if file_ext == ".pdf":
                            logger.info("Attempting PyPDF text extraction fallback")
                            try:
                                from pypdf import PdfReader
                                reader = PdfReader(file_path)
                                extracted_pages = []
                                for page in reader.pages:
                                    try:
                                        extracted_pages.append(page.extract_text() or "")
                                    except Exception as per_page_err:
                                        logger.info(f"Warning: failed to extract a page: {per_page_err}")
                                        extracted_pages.append("")
                                file_text = "\n\n".join(extracted_pages).strip()
                            except Exception as pypdf_error:
                                logger.info(f"PyPDF fallback failed: {pypdf_error}")
                                file_text = None

                        if not file_text:
                            logger.info("Falling back to raw text read of file contents")
                            try:
                                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                    file_text = f.read()
                            except Exception as read_error:
                                logger.info(f"Fallback read failed: {read_error}")
                                file_text = ""

                        if file_text and file_text.strip():
                            docs = [Document(
                                page_content=file_text,
                                metadata={
                                    "source": source_name,
                                    "file_path": file_path,
                                    "filename": os.path.basename(file_path),
                                }
                            )]
                        else:
                            logger.info("Creating a simple document as fallback (no text extracted)")
                            docs = [Document(
                                page_content=f"Document: {os.path.basename(file_path)}",
                                metadata={
                                    "source": source_name,
                                    "file_path": file_path,
                                    "filename": os.path.basename(file_path),
                                }
                            )]

                    for doc in docs:
                        if not doc.metadata:
                            doc.metadata = {}

                        cleaned_metadata = {}
                        cleaned_metadata["source"] = source_name
                        cleaned_metadata["file_path"] = file_path
                        cleaned_metadata["filename"] = os.path.basename(file_path)

                        for key, value in doc.metadata.items():
                            if key not in ["source", "file_path"]:
                                if isinstance(value, (list, dict, set)):
                                    cleaned_metadata[key] = str(value)
                                elif value is not None:
                                    cleaned_metadata[key] = str(value)

                        doc.metadata = cleaned_metadata
                    documents.extend(docs)
                    logger.debug({
                        "message": "Loaded documents from file",
                        "file_path": file_path,
                        "document_count": len(docs)
                    })
                except Exception as e:
                    logger.error({
                        "message": "Error loading file",
                        "file_path": file_path,
                        "error": str(e)
                    }, exc_info=True)
                    continue

            logger.info(f"Total documents loaded: {len(documents)}")
            return documents

        except Exception as e:
            logger.error({
                "message": "Error loading documents",
                "error": str(e)
            }, exc_info=True)
            raise

    def index_documents(self, documents: List[Document]) -> None:
        """Index documents by generating embeddings and storing them in Geode."""
        try:
            logger.debug({
                "message": "Starting document indexing",
                "document_count": len(documents)
            })

            splits = self.text_splitter.split_documents(documents)
            logger.debug({
                "message": "Split documents into chunks",
                "chunk_count": len(splits)
            })

            texts = [doc.page_content for doc in splits]
            embeddings = self.embeddings.embed_documents(texts)

            import asyncio
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

            if loop and loop.is_running():
                # We are inside an async context; schedule as a coroutine
                asyncio.ensure_future(self._store_documents(splits, embeddings))
            else:
                asyncio.run(self._store_documents(splits, embeddings))

            self.flush_store()

            logger.debug({
                "message": "Document indexing completed"
            })
        except Exception as e:
            logger.error({
                "message": "Error during document indexing",
                "error": str(e)
            }, exc_info=True)
            raise

    async def _ensure_index(self, conn) -> None:
        """Create the HNSW vector index for RagDocument embeddings (idempotent)."""
        if self._index_ensured:
            return
        try:
            await conn.execute(
                "CREATE INDEX rag_emb_idx ON RagDocument(embedding) "
                "USING vector WITH (metric: 'cosine')"
            )
            logger.debug({"message": "Created HNSW index rag_emb_idx"})
        except Exception as e:
            msg = str(e)
            if "42P07" in msg or "already exists" in msg.lower() or "duplicate" in msg.lower():
                pass  # index already exists
            else:
                raise
        self._index_ensured = True

    async def _store_documents(self, splits: List[Document], embeddings: List[List[float]]) -> None:
        """Store document chunks with embeddings in Geode as RagDocument nodes."""
        async with self.client.connection() as conn:
            await self._ensure_index(conn)
            for doc, embedding in zip(splits, embeddings):
                doc_id = str(uuid.uuid4())

                await conn.execute(
                    "CREATE (d:RagDocument {"
                    "docId: $docId, "
                    "text: $text, "
                    "source: $source, "
                    "embedding: vectorf32(toString($vec))"
                    "})",
                    {
                        "docId": doc_id,
                        "text": doc.page_content,
                        "source": doc.metadata.get("source", "unknown"),
                        "vec": _vec_param(embedding),
                    }
                )

    def flush_store(self):
        """No-op for Geode. Geode persists data immediately."""
        logger.debug({
            "message": "Geode store flush (no-op, data is persisted automatically)"
        })

    def get_documents(self, query: str, k: int = 8, sources: List[str] = None) -> List[Document]:
        """Get relevant documents using knn vector similarity search in Geode."""
        try:
            query_embedding = self.embeddings.embed_query(query)

            logger.debug({
                "message": "Retrieving documents from Geode",
                "query": query[:100],
                "k": k,
                "sources": sources
            })

            import asyncio
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    docs = pool.submit(
                        asyncio.run, self._query_documents(query_embedding, k, sources)
                    ).result()
            else:
                docs = asyncio.run(self._query_documents(query_embedding, k, sources))

            logger.debug({
                "message": "Retrieved documents",
                "query": query[:100],
                "document_count": len(docs)
            })

            return docs
        except Exception as e:
            logger.error({
                "message": "Error retrieving documents",
                "error": str(e)
            }, exc_info=True)
            return []

    async def _query_documents(
        self, query_embedding: List[float], k: int, sources: List[str] = None
    ) -> List[Document]:
        """Run a knn similarity query (with full-scan fallback) and return documents.

        vector_cosine returns SIMILARITY (self = 1), so higher is better.
        Scores can arrive as decimal strings — projected via toString() and
        coerced with float(); final ranking is done client-side on the floats.
        """
        params = {"vec": _vec_param(query_embedding), "k": int(k)}

        source_filter = ""
        if sources:
            placeholders = []
            for i, source in enumerate(sources):
                params[f"src{i}"] = source
                placeholders.append(f"$src{i}")
            source_filter = f" AND d.source IN [{', '.join(placeholders)}]"

        knn_query = (
            "MATCH (d:RagDocument) "
            "WHERE id(d) IN knn('embedding', vectorf32(toString($vec)), $k, 'cosine') "
            "WITH d, vector_cosine(d.embedding, vectorf32(toString($vec))) AS score "
            f"WHERE score > 0.1{source_filter} "
            "RETURN d.text AS text, d.source AS source, toString(score) AS score"
        )
        scan_query = (
            "MATCH (d:RagDocument) "
            "WHERE d.embedding IS NOT NULL "
            "WITH d, vector_cosine(d.embedding, vectorf32(toString($vec))) AS score "
            f"WHERE score > 0.1{source_filter} "
            "RETURN d.text AS text, d.source AS source, toString(score) AS score "
            "LIMIT $k"
        )

        async with self.client.connection() as conn:
            await self._ensure_index(conn)
            try:
                page, _ = await conn.query(knn_query, params)
            except QueryError as e:
                # Falling back to a full scan is intentionally resilient (e.g. the
                # HNSW index may not be built yet right after startup, which is
                # benign and self-resolves once _ensure_index()/indexing completes).
                # Log at warning so an unexpected regression in the knn fast path
                # (e.g. a real query/index bug) doesn't silently degrade to a full
                # scan on every request without being noticed.
                logger.warning({
                    "message": "knn fast path unavailable, falling back to full scan",
                    "error": str(e)
                })
                page, _ = await conn.query(scan_query, params)

            docs = []
            for row in page.rows:
                text = _val(row, "text", "")
                source = _val(row, "source", "unknown")
                score_raw = _val(row, "score", "0.0")
                try:
                    score = float(score_raw) if score_raw else 0.0
                except (TypeError, ValueError):
                    score = 0.0
                docs.append(Document(
                    page_content=text,
                    metadata={"source": source, "score": score}
                ))
            # Client-side sort is the ranking authority, not just a safety net:
            # GQL-side ORDER BY score was dropped because float projections
            # (toString(score)) decode as NULL through ORDER BY, which would
            # silently scramble ranking. Re-sort here on the parsed floats and
            # slice to k to enforce the intended similarity ordering.
            docs.sort(key=lambda d: d.metadata["score"], reverse=True)
            return docs[:k]

    def delete_collection(self, collection_name: str) -> bool:
        """Delete all RagDocument nodes matching a source name.

        Args:
            collection_name: Source name of documents to delete

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import asyncio
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    pool.submit(asyncio.run, self._delete_source(collection_name)).result()
            else:
                asyncio.run(self._delete_source(collection_name))

            if self.on_source_deleted:
                self.on_source_deleted(collection_name)

            logger.debug({
                "message": "Collection deleted successfully",
                "collection_name": collection_name
            })
            return True
        except Exception as e:
            logger.error({
                "message": "Error deleting collection",
                "collection_name": collection_name,
                "error": str(e)
            }, exc_info=True)
            return False

    async def _delete_source(self, source_name: str) -> None:
        """Delete all RagDocument nodes with a given source."""
        async with self.client.connection() as conn:
            await conn.execute(
                "MATCH (d:RagDocument {source: $source}) DELETE d",
                {"source": source_name}
            )


def create_vector_store_with_config(
    config_manager,
    host: str = None,
    port: int = None,
    username: str = None,
    password: str = None
) -> GeodeVectorStore:
    """Factory function to create a GeodeVectorStore with ConfigManager integration.

    Args:
        config_manager: ConfigManager instance for source management
        host: Geode server host (defaults to GEODE_HOST env var or 'geode')
        port: Geode server port (defaults to GEODE_PORT env var or 3141)
        username: Geode auth username (defaults to GEODE_USERNAME env var)
        password: Geode auth password (defaults to GEODE_PASSWORD env var)

    Returns:
        GeodeVectorStore instance with source deletion callback
    """
    if host is None:
        host = os.getenv("GEODE_HOST", "geode")
    if port is None:
        port = int(os.getenv("GEODE_PORT", "3141"))
    if username is None:
        username = os.getenv("GEODE_USERNAME", "")
    if password is None:
        password = os.getenv("GEODE_PASSWORD", "")

    def handle_source_deleted(source_name: str):
        """Handle source deletion by updating config."""
        config = config_manager.read_config()
        if hasattr(config, 'sources') and source_name in config.sources:
            config.sources.remove(source_name)
            config_manager.write_config(config)

    return GeodeVectorStore(
        host=host,
        port=port,
        username=username,
        password=password,
        on_source_deleted=handle_source_deleted
    )
