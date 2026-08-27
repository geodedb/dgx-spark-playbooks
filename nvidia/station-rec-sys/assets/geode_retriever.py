"""Geode vector retrieval backend for the HLLM recommender.

FAISS-equivalent top-K inner-product retrieval backed by Geode's native
VectorF32 type + HNSW vector index. Loads the item embeddings produced by
``extract_embeddings.py`` (hllm_item_embeddings.npy + hllm_item_id_map.npy)
into Geode as ``RecItem`` nodes (idempotent MERGE on itemId) and serves
``search(query, k)`` via Geode's ``knn()`` with the ``'dot'`` metric —
the same inner-product ranking as ``faiss.IndexFlatIP``.

Requirements:
  - geode-client >= v0.5.0 (installed by assets/setup.sh; raw float score
    projections and server-side ORDER BY need v0.5.0+).
  - A running Geode server built from geode:dev. The HLLM embedding
    dimension is 2048, which exceeds the 1792-dim VectorF32 cap of the
    v0.8.5 release; geode:dev raises the cap to 65535.
    (registry.gitlab.com/devnw/codepros/geode/geode:dev)

Configuration (environment):
  GEODE_URL       Server DSN, e.g. quic://127.0.0.1:3141 (default) or a
                  bare host:port. grpc:// URLs select the gRPC transport.
  GEODE_USERNAME  Auth username (optional).
  GEODE_PASSWORD  Auth password (optional).

Storage pattern (proven in nvidia/multi-agent-chatbot geode_vector_store.py):
embeddings are param-bound as JSON strings and stored with
``vectorf32(toString($vec))`` — knn()/HNSW ignore LIST-typed properties,
so true VectorF32 storage is required.
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from geode_client import Client, QueryError, TransportType

LABEL = "RecItem"
INDEX_NAME = "rec_emb_idx"
METRIC = "dot"  # accepted spellings: 'dot', 'dot_product', 'inner_product'

# Geode periodically runs storage maintenance (WAL checkpoint) that can hold
# the graph write lock for tens of seconds once vector-heavy data accumulates;
# a write issued in that window blocks until it finishes. A generous read
# timeout plus reconnect-retry keeps the bulk load resilient to those stalls.
READ_TIMEOUT_S = 300.0
LOAD_RETRIES = 3


def _vec_param(embedding: Sequence[float]) -> str:
    """Serialize an embedding for param-bound VectorF32 storage."""
    return json.dumps([float(x) for x in embedding])


def _val(row: dict, key: str, default=None):
    """Extract a plain Python value from a Geode row entry."""
    v = row.get(key)
    if v is None:
        return default
    if isinstance(v, (str, int, float, bool)):
        return v
    if hasattr(v, "as_string"):
        return v.as_string
    return default


def _run(coro):
    """Run a coroutine from sync code, tolerating a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class GeodeRetriever:
    """Inner-product top-K retrieval over RecItem nodes in Geode."""

    def __init__(
        self,
        url: str = "quic://127.0.0.1:3141",
        username: str = "",
        password: str = "",
        dim: Optional[int] = None,
    ):
        self.url = url if "://" in url else f"quic://{url}"
        self.username = username
        self.password = password
        self.dim = dim
        self.client = self._make_client()
        self._index_ensured = False

    @classmethod
    def from_env(cls, dim: Optional[int] = None) -> "GeodeRetriever":
        return cls(
            url=os.environ.get("GEODE_URL", "quic://127.0.0.1:3141"),
            username=os.environ.get("GEODE_USERNAME", ""),
            password=os.environ.get("GEODE_PASSWORD", ""),
            dim=dim,
        )

    def _make_client(self) -> Client:
        base = Client.from_url(self.url)
        kwargs = dict(base.config)
        kwargs.pop("host", None)
        kwargs.pop("port", None)
        kwargs.pop("transport_type", None)
        kwargs["skip_verify"] = True
        kwargs["read_timeout"] = READ_TIMEOUT_S
        if self.username:
            kwargs["username"] = self.username
        if self.password:
            kwargs["password"] = self.password
        return Client(
            host=base.host,
            port=base.port,
            transport_type=base.transport_type,
            **kwargs,
        )

    # -- schema ---------------------------------------------------------

    async def _ensure_index(self, conn) -> None:
        """Create the HNSW inner-product index (idempotent)."""
        if self._index_ensured:
            return
        try:
            await conn.execute(
                f"CREATE INDEX {INDEX_NAME} ON {LABEL}(emb) "
                f"USING vector WITH (metric: '{METRIC}')"
            )
        except Exception as e:  # noqa: BLE001 - idempotency probe
            msg = str(e).lower()
            if "already exists" not in msg and "duplicate" not in msg and "42p07" not in msg:
                raise
        self._index_ensured = True

    async def _ensure_key_index(self, conn) -> None:
        """B-tree index on itemId so MERGE lookups don't do an O(n) label
        scan that hydrates every stored embedding per statement."""
        try:
            await conn.execute(f"CREATE INDEX rec_itemid_idx ON {LABEL}(itemId)")
        except Exception as e:  # noqa: BLE001 - idempotency probe
            msg = str(e).lower()
            if "already exists" not in msg and "duplicate" not in msg and "42p07" not in msg:
                raise

    # -- load path ------------------------------------------------------

    def load_items(
        self,
        embeddings: np.ndarray,
        id_map: Sequence[str],
        titles: Optional[Dict[str, str]] = None,
        pad_token: str = "[PAD]",
        progress_every: int = 2000,
    ) -> int:
        """Idempotently load item embeddings into Geode (MERGE on itemId).

        ``embeddings`` is the (N, dim) matrix from hllm_item_embeddings.npy;
        ``id_map`` the aligned item-id array from hllm_item_id_map.npy (row 0
        is HLLM padding and is skipped). ``hllmIdx`` stores each item's row
        index in the .npy so search results map straight back into the same
        index space FAISS uses.

        Returns the number of item rows resident in Geode after the call.
        """
        rows = [
            (i, str(iid)) for i, iid in enumerate(id_map) if str(iid) != pad_token
        ]
        return int(_run(self._load_items(embeddings, rows, titles or {}, progress_every)))

    async def _load_items(self, embeddings, rows, titles, progress_every) -> int:
        conn = await self.client.connect()
        try:
            # Fast path: everything already loaded (idempotent re-launch).
            page, _ = await conn.query(
                f"MATCH (i:{LABEL}) RETURN count(i) AS n"
            )
            existing = 0
            if page.rows:
                try:
                    existing = int(float(_val(page.rows[0], "n", 0) or 0))
                except (TypeError, ValueError):
                    existing = 0
            if existing == len(rows):
                print(f"  Geode already holds {existing:,} {LABEL} nodes; skipping load")
                await self._ensure_index(conn)
                return existing

            # NOTE: the HNSW index is created AFTER the bulk load, not before.
            # Geode rebuilds a vector index from scratch on every write to the
            # indexed property, so inserting N items with the index already in
            # place costs O(N^2 * dim); loading first and indexing once keeps
            # the load linear.
            print(f"  Loading {len(rows):,} items into Geode ({existing:,} present)...")
            await self._ensure_key_index(conn)
            for done, (idx, item_id) in enumerate(rows, 1):
                query = (
                    f"MERGE (i:{LABEL} {{itemId: $itemId}}) "
                    "SET i.title = $title, i.hllmIdx = $idx, "
                    "i.emb = vectorf32(toString($vec))"
                )
                params = {
                    "itemId": item_id,
                    "title": str(titles.get(item_id, ""))[:512],
                    "idx": int(idx),
                    "vec": _vec_param(embeddings[idx]),
                }
                for attempt in range(LOAD_RETRIES + 1):
                    try:
                        await conn.execute(query, params)
                        break
                    except QueryError:
                        raise
                    except Exception as e:  # transport timeout / dropped conn
                        if attempt >= LOAD_RETRIES:
                            raise
                        print(f"  [geode_retriever] write stalled at item {done:,} "
                              f"({type(e).__name__}); reconnecting "
                              f"(attempt {attempt + 1}/{LOAD_RETRIES})")
                        try:
                            await conn.close()
                        except Exception:
                            pass
                        conn = await self.client.connect()
                if progress_every and done % progress_every == 0:
                    print(f"    {done:,}/{len(rows):,} items loaded")

            print("  Building HNSW index (metric 'dot')...")
            await self._ensure_index(conn)
            return len(rows)
        finally:
            try:
                await conn.close()
            except Exception:
                pass

    # -- query path -----------------------------------------------------

    def top_k(self, user_vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """Return the top-``k`` items by inner product with ``user_vec``.

        Result: list of ``(hllm_idx, score)`` sorted by descending score —
        the Geode-native equivalent of ``faiss.IndexFlatIP.search``.
        """
        vec = np.asarray(user_vec, dtype=np.float32).reshape(-1)
        return _run(self._top_k(vec.tolist(), int(k)))

    async def _top_k(self, vec: List[float], k: int) -> List[Tuple[int, float]]:
        # Candidate selection uses knn(..., 'dot') — true inner-product HNSW
        # ordering. The score projection uses vector_cosine, NOT vector_dot:
        # the engine's vector_dot truncates its result to an integer
        # (src/eval_extended.zig vectorDot returns .Int), which collapses
        # unit-vector dot products to 0. HLLM item and user embeddings are
        # L2-normalized (extract_embeddings.py / app.py), so cosine == inner
        # product exactly for this pipeline.
        params = {"vec": _vec_param(vec), "k": k}
        knn_query = (
            f"MATCH (i:{LABEL}) "
            f"WHERE id(i) IN knn('emb', vectorf32(toString($vec)), $k, '{METRIC}') "
            "WITH i, vector_cosine(i.emb, vectorf32(toString($vec))) AS score "
            "RETURN i.hllmIdx AS idx, score "
            "ORDER BY score DESC"
        )
        scan_query = (
            f"MATCH (i:{LABEL}) "
            "WHERE i.emb IS NOT NULL "
            "WITH i, vector_cosine(i.emb, vectorf32(toString($vec))) AS score "
            "RETURN i.hllmIdx AS idx, score "
            "ORDER BY score DESC "
            "LIMIT $k"
        )
        async with self.client.connection() as conn:
            await self._ensure_index(conn)
            try:
                page, _ = await conn.query(knn_query, params)
            except QueryError as e:
                # The HNSW index may not be built yet right after startup;
                # degrade to a full scan rather than failing the request.
                print(f"  [geode_retriever] knn unavailable, full-scan fallback: {e}")
                page, _ = await conn.query(scan_query, params)

            out: List[Tuple[int, float]] = []
            for row in page.rows:
                try:
                    idx = int(float(_val(row, "idx", -1)))
                    score = float(_val(row, "score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if idx >= 0:
                    out.append((idx, score))
            return out[:k]

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """FAISS-compatible search: (1, dim) query -> (distances, indices).

        Both returned arrays have shape (1, k); missing slots (catalog
        smaller than k) are padded with -1 indices, matching FAISS.
        """
        q = np.asarray(query, dtype=np.float32)
        if q.ndim == 2:
            if q.shape[0] != 1:
                raise ValueError("GeodeRetriever.search supports one query row per call")
            q = q[0]
        hits = self.top_k(q, k)
        indices = np.full((1, k), -1, dtype=np.int64)
        distances = np.full((1, k), -np.inf, dtype=np.float32)
        for j, (idx, score) in enumerate(hits[:k]):
            indices[0, j] = idx
            distances[0, j] = score
        return distances, indices
