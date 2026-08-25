#!/usr/bin/env python3
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
"""Headless knowledge-graph ingestion driver for geode v0.8.5.

Ingests a directory of .md files into a geode graph database:

  for each file:
    chunk (recursive splitter, --chunk-size/--chunk-overlap,
    default 2000 chars / 200 overlap)
    -> embed chunks via an OpenAI-compatible /v1/embeddings endpoint
       (batched, 32 texts/request)
    -> MERGE DocumentChunk {docId, seq, text, source, emb} nodes
    -> extract [subject, predicate, object] triples per chunk via an
       OpenAI-compatible /v1/chat/completions endpoint (strict-JSON prompt)
    -> MERGE Entity nodes, then MATCH+MERGE RELATIONSHIP {type} edges
    -> MERGE a ProcessedDocument marker (idempotent resume)

Run it with the multi-agent-chatbot backend venv, which carries
geode_client v0.4.1 and requests:

  uv run --project nvidia/multi-agent-chatbot/assets/backend \
      python nvidia/txt2kg/assets/scripts/ingest_kb.py \
      --dir /path/to/kb-md --limit 10 --resume

Env:
  VLLM_API_KEY        Bearer token for the chat-completions endpoint (required
                      unless --skip-triples).
  VLLM_API_URL        Override the chat-completions URL.
  EMBEDDINGS_API_URL  Override the embeddings URL.
  GEODE_ALLOW_INSECURE_TLS=1  (informational; the client always uses
                      skip_verify=True for the self-signed dev server).

geode v0.8.5 client rules honored here (copied from the migrated
geode_vector_store.py):
  - param binding everywhere;
  - embeddings stored as `SET ... embedding: vectorf32(toString($vec))` with
    the vector serialized as a JSON string ("_vec_param");
  - floats projected via toString() when read back;
  - edges are never written with chained MERGE: nodes are MERGEd first, the
    edge statement is a separate MATCH + MERGE.

ENG-3911 workaround: every embedding (documents and queries alike) is
truncated to its first EMBED_DIM=1536 dims and L2-renormalized in
shrink_embedding(), and floats are serialized at 7 significant digits —
see the EMBED_DIM comment below. The chunk vector is stored in property
`emb` because DocumentChunk.embedding's dimension was pinned at 2560 by a
pre-fix write attempt and v0.8.5 cannot reset it. A `--search "some
question"` mode runs a validation knn query through the identical
embedding path.

Chunk embeddings are stored for ALL chunks. Entity embeddings are
deliberately skipped: embedding text equal to the bare entity name is
low-value at corpus scale, and the entity_emb_idx HNSW index tolerates
nodes without the embedding property.
"""

import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:  # pragma: no cover - urllib3 always ships with requests
    pass

warnings.filterwarnings("ignore", message=".*skip_verify.*")

from geode_client import Client, QueryError, TransportType  # noqa: E402
from geode_client import transport as _geode_transport  # noqa: E402

# The QUIC client's per-message read timeout defaults to a hardcoded 30s,
# while the v0.8.5 dev server can stall for minutes under vector-index write
# load and a knn query over a warm corpus can exceed 30s. Widen only the
# DEFAULT (call sites that pass an explicit timeout, e.g. the handshake,
# keep their own values).
DB_READ_TIMEOUT = 180.0

_orig_read_message = _geode_transport.QuicTransport._read_message


def _read_message_patched(self, read_timeout: float = DB_READ_TIMEOUT):
    return _orig_read_message(self, read_timeout=read_timeout)


_geode_transport.QuicTransport._read_message = _read_message_patched

# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDINGS_URL = "https://celebrimbor.shire-altair.ts.net:8001/v1/embeddings"
# Encoder model id; override with env EMBEDDINGS_MODEL_ID on encoder swaps
# (e.g. Qwen/Qwen3-Embedding-0.6B for throughput-bound full-corpus runs).
DEFAULT_EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL_ID", "Qwen/Qwen3-Embedding-4B")
DEFAULT_LLM_URL = "http://curumo.shire-altair.ts.net:8000/v1/chat/completions"
DEFAULT_LLM_MODEL = "Qwen/Qwen3.8-27B"

# ENG-3911: geode v0.8.5 caps property VALUE size (~8KB). vectorf32 storage
# works at <=1792 dims and fails with `58000 PropertyKeyTooLong` at >=2040;
# the encoder (Qwen3-Embedding-4B) emits 2560-dim vectors. Every embedding —
# documents AND queries alike — is therefore truncated to its first EMBED_DIM
# dims and L2-renormalized. Qwen3-Embedding is Matryoshka-trained, so prefix
# truncation is the sanctioned degradation. 1536 f32 = 6144 bytes, comfortably
# under the cap. Remove once ENG-3911 lands.
#
# The chunk vector lives in property `emb`, NOT `embedding`: geode pins a
# vector property's dimension per (table, column) on the FIRST write attempt,
# and the pre-fix run pinned DocumentChunk.embedding at 2560 before the size
# check rejected the value. v0.8.5 has no DROP TABLE / ALTER TABLE DDL to
# reset it (verified live), so the driver uses a fresh column instead.
# Stored vector width. Override with env EMBED_DIM when the encoder's native
# dimension differs (e.g. EMBED_DIM=1024 for Qwen3-Embedding-0.6B, which needs
# no truncation at all — it is already under the ~1792-dim cap).
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))
EMB_PROP = "emb"
CHUNK_INDEX = "chunk_emb_idx"

CHUNK_SIZE = 2000  # default; override with --chunk-size
CHUNK_OVERLAP = 200  # default; override with --chunk-overlap
EMBED_BATCH_SIZE = 32
# Floor for embed-input truncation. The effective limit is
# max(EMBED_TRUNCATE_CHARS, chunk_size + chunk_overlap) so a larger
# --chunk-size can never be silently cut mid-chunk (chunks may carry up to
# `overlap` extra prefix chars). 6300 chars ~= 1600 tokens, far under the
# encoder's 8192-token ctx.
EMBED_TRUNCATE_CHARS = 6000
ENTITY_NAME_MAX = 512
LLM_MAX_TOKENS = 2000
LLM_TEMPERATURE = 0.1
HTTP_TIMEOUT = 300
MAX_RETRIES = 5
# The dev server occasionally stalls for minutes (background work?) while the
# QUIC client's read timeout is a hardcoded 30s; ride the stall out with a
# generous retry budget (worst case ~6.5 min of backoff). MERGE semantics make
# replaying a partially applied batch safe.
DB_MAX_RETRIES = 6
DB_BACKOFF_CAP = 120

TRIPLE_SYSTEM_PROMPT = (
    "Extract subject-predicate-object triples from the text. Reply ONLY with "
    "a JSON array of [subject, predicate, object] string triples, no prose."
)

LOG_PATH = Path.cwd() / "ingest_kb.log"
FAILURES_PATH = Path.cwd() / "ingest_failures.jsonl"
LOCK_PATH = Path(__file__).resolve().parent / "ingest_kb.pid"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    try:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _pid_is_live_ingest(pid: int) -> bool:
    """True iff `pid` is a live process whose cmdline mentions ingest_kb.py."""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False  # no such process (or no /proc entry) -> stale
    return b"ingest_kb.py" in cmdline


def acquire_lock() -> None:
    """PID lockfile guard (ENG-3914): overlapping ingest processes create
    MERGE duplicates via late-committing stalled writes, so refuse to start
    while another ingest_kb.py is alive. A stale lockfile (dead PID, or PID
    reused by an unrelated process) is safe to replace."""
    if LOCK_PATH.exists():
        try:
            old_pid = int(LOCK_PATH.read_text().strip())
        except (OSError, ValueError):
            old_pid = -1
        if old_pid > 0 and _pid_is_live_ingest(old_pid):
            print(
                f"error: another ingest_kb.py (pid {old_pid}) is already running "
                f"per lockfile {LOCK_PATH}; overlapping ingest processes create "
                "MERGE duplicates (ENG-3914). Refusing to start.",
                file=sys.stderr,
            )
            sys.exit(3)
        log(f"WARN replacing stale lockfile {LOCK_PATH} (pid {old_pid} not a live ingest)")
    LOCK_PATH.write_text(str(os.getpid()))


def release_lock() -> None:
    try:
        if LOCK_PATH.exists() and LOCK_PATH.read_text().strip() == str(os.getpid()):
            LOCK_PATH.unlink()
    except OSError:
        pass


def record_failure(file: str, stage: str, error: str) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file": file,
        "stage": stage,
        "error": error[:2000],
    }
    try:
        with FAILURES_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass
    log(f"FAIL file={file} stage={stage} error={error[:300]}")


# ---------------------------------------------------------------------------
# Chunking (simple recursive splitter; no langchain dependency)
# ---------------------------------------------------------------------------

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_recursive(text: str, size: int, seps: list) -> list:
    """Split text into pieces no longer than `size`, preferring the earliest
    separator in `seps` that produces a useful split."""
    if len(text) <= size:
        return [text]
    sep = seps[0] if seps else ""
    rest = seps[1:] if seps else []
    if sep == "":
        return [text[i : i + size] for i in range(0, len(text), size)]
    parts = text.split(sep)
    if len(parts) == 1:
        return _split_recursive(text, size, rest)
    pieces = []
    buf = ""
    for part in parts:
        candidate = buf + sep + part if buf else part
        if len(candidate) <= size:
            buf = candidate
            continue
        if buf:
            pieces.append(buf)
        if len(part) > size:
            pieces.extend(_split_recursive(part, size, rest))
            buf = ""
        else:
            buf = part
    if buf:
        pieces.append(buf)
    return pieces


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Recursive character split with overlap; returns non-empty chunks."""
    pieces = _split_recursive(text, size, _SEPARATORS)
    chunks = []
    prev_tail = ""
    for piece in pieces:
        chunk = (prev_tail + piece) if prev_tail else piece
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        prev_tail = piece[-overlap:] if overlap and len(piece) > overlap else ""
    return chunks


# ---------------------------------------------------------------------------
# Entity / triple normalization
# ---------------------------------------------------------------------------


def normalize_entity(name) -> str:
    if not isinstance(name, str):
        return ""
    name = re.sub(r"\s+", " ", name).strip()
    return name[:ENTITY_NAME_MAX]


_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", re.MULTILINE)


def parse_triples(content: str) -> list:
    """Parse an LLM reply into a list of (s, p, o) string triples.

    Defensive: strips code fences and surrounding prose, tolerates junk
    entries. Raises ValueError when no JSON array can be recovered.
    """
    if not content:
        raise ValueError("empty content")
    text = _FENCE_RE.sub("", content.strip()).strip()
    data = None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            slice_ = text[start : end + 1]
            try:
                data = json.loads(slice_)
            except (json.JSONDecodeError, ValueError):
                # Replies that hit the max_tokens budget arrive truncated
                # mid-array; closing after the last complete inner triple
                # salvages everything before the cut.
                data = json.loads(slice_ + "]")
        else:
            raise ValueError("no JSON array found in reply")
    if not isinstance(data, list):
        raise ValueError("reply is not a JSON array")
    triples = []
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        s, p, o = (normalize_entity(item[0]), normalize_entity(item[1]), normalize_entity(item[2]))
        if s and p and o:
            triples.append((s, p, o))
    return triples


# ---------------------------------------------------------------------------
# HTTP helpers (requests, run in threads; retried with backoff)
# ---------------------------------------------------------------------------


def _post_with_retry(url: str, payload: dict, headers: dict, verify: bool) -> dict:
    """POST with retry+backoff for transient failures only.

    Retryable: HTTP 429, 5xx, connection/timeout errors. Non-retryable
    client errors (400/401/403/404, any other 4xx) fail fast — retrying a
    bad request or bad credentials just burns the retry budget.
    """
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                url, json=payload, headers=headers, verify=verify, timeout=HTTP_TIMEOUT
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            elif resp.status_code >= 400:
                raise RuntimeError(
                    f"POST {url} failed with non-retryable "
                    f"HTTP {resp.status_code}: {resp.text[:200]}"
                )
            else:
                return resp.json()
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"POST {url} failed after {MAX_RETRIES} attempts: {last_err}")


def shrink_embedding(vec: list) -> list:
    """Truncate to the first EMBED_DIM dims and L2-renormalize (ENG-3911).

    Applied identically to document and query embeddings so cosine
    similarities remain comparable. Zero-norm slices are returned as zeros
    rather than dividing by zero.

    No-op when the vector already fits: an encoder whose native dimension is
    <= EMBED_DIM (e.g. 1024-dim Qwen3-Embedding-0.6B) emits unit-norm vectors
    that need neither truncation nor renormalization.
    """
    if len(vec) <= EMBED_DIM:
        return list(vec)
    v = vec[:EMBED_DIM]
    norm = math.sqrt(sum(x * x for x in v))
    if norm <= 0.0:
        return [0.0] * len(v)
    return [x / norm for x in v]


class Embedder:
    def __init__(self, url: str, model: str, truncate_chars: int = EMBED_TRUNCATE_CHARS):
        self.url = url
        self.model = model
        self.truncate_chars = max(truncate_chars, EMBED_TRUNCATE_CHARS)

    def embed_batch(self, texts: list) -> list:
        truncated = [t[: self.truncate_chars] for t in texts]
        data = _post_with_retry(
            self.url,
            {"model": self.model, "input": truncated},
            {"Content-Type": "application/json"},
            verify=False,  # self-signed tailscale cert
        )
        rows = sorted(data["data"], key=lambda d: d.get("index", 0))
        # ENG-3911: shrink at the single point every embedding flows through,
        # so ingestion and the query side can never diverge.
        return [shrink_embedding(row["embedding"]) for row in rows]


class TripleExtractor:
    def __init__(self, url: str, model: str, api_key: str):
        self.url = url
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def _call(self, chunk: str, temperature: float) -> str:
        data = _post_with_retry(
            self.url,
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": TRIPLE_SYSTEM_PROMPT},
                    {"role": "user", "content": chunk},
                ],
                "max_tokens": LLM_MAX_TOKENS,
                "temperature": temperature,
                # Disable Qwen3 thinking: with reasoning enabled the model
                # spends the entire max_tokens budget on the reasoning field
                # and returns empty/truncated content (verified live).
                "chat_template_kwargs": {"enable_thinking": False},
            },
            self.headers,
            verify=True,
        )
        message = data["choices"][0]["message"]
        # The server may include a "reasoning" field — ignore it, use content.
        return message.get("content") or ""

    def extract(self, chunk: str) -> list:
        try:
            return parse_triples(self._call(chunk, LLM_TEMPERATURE))
        except (ValueError, KeyError, json.JSONDecodeError):
            # one retry at temperature 0, then let the caller record failure
            return parse_triples(self._call(chunk, 0.0))


# ---------------------------------------------------------------------------
# Geode helpers
# ---------------------------------------------------------------------------


def _vec_param(embedding: list) -> str:
    """Serialize an embedding for param-bound VectorF32 storage.

    The only fully param-bound form geode v0.8.5 accepts is
    `vectorf32(toString($vec))` with $vec bound to the JSON string of the
    float list; LIST-typed properties are ignored by knn()/HNSW.

    Floats are rounded to 7 significant digits (full f32 precision) to keep
    the serialized toString payload well under the v0.8.5 property-size cap
    (ENG-3911).
    """
    return json.dumps([float(f"{x:.6e}") for x in embedding])


def _val(row: dict, key: str, default=None):
    v = row.get(key)
    if v is None:
        return default
    if isinstance(v, (str, int, float, bool)):
        return v
    if hasattr(v, "as_string"):
        return v.as_string
    return default


async def preflight_vector_write(client: Client) -> None:
    """Prove an EMBED_DIM vectorf32 chunk property can actually be stored.

    The first sample run burned ~21 minutes of embedding before the very
    first chunk write failed with `58000 PropertyKeyTooLong` (ENG-3911).
    This writes and deletes a throwaway probe node up front — through the
    REAL DocumentChunk table and EMB_PROP column, so both the value-size cap
    and the per-column established-dimension pin are exercised — and then
    ensures the HNSW index exists. A regression fails in seconds, not after
    the embedding pass.
    """
    probe = [1.0] + [0.0] * (EMBED_DIM - 1)
    async with client.connection() as conn:
        await conn.execute(
            "MERGE (c:DocumentChunk {docId: $docId, seq: $seq}) "
            f"SET c.{EMB_PROP} = vectorf32(toString($vec))",
            {"docId": "__preflight__", "seq": -1, "vec": _vec_param(probe)},
        )
        if os.getenv("INGEST_SKIP_INDEX") == "1":
            # Index-free bulk load mode: HNSW insert maintenance is the v0.8.5
            # write bottleneck (multi-minute rwlock stalls, ~15-20 chunk
            # MERGEs/min). With the index dropped, bulk MERGE throughput
            # recovers; run CREATE INDEX once after the load (backfill):
            #   CREATE INDEX chunk_emb_idx ON DocumentChunk(emb)
            #     USING vector WITH (metric: 'cosine')
            log(f"preflight: INGEST_SKIP_INDEX=1 -> not ensuring {CHUNK_INDEX} "
                "(index-free bulk load; CREATE INDEX must be run post-load)")
        else:
            try:
                await conn.execute(
                    f"CREATE INDEX {CHUNK_INDEX} ON DocumentChunk({EMB_PROP}) "
                    "USING vector WITH (metric: 'cosine')"
                )
            except QueryError as e:
                msg = str(e).lower()
                if "already exists" not in msg and "duplicate" not in msg and "42p07" not in msg:
                    raise
        await conn.execute(
            "MATCH (c:DocumentChunk {docId: $docId}) DETACH DELETE c",
            {"docId": "__preflight__"},
        )


async def fetch_marker_statuses(client: Client) -> dict:
    """Return {docId: status} for every ProcessedDocument marker.

    Mode-aware resume semantics (see task-9 report, "--skip-triples marker
    semantics"): an embeddings-only run writes status='embedded'; a
    triples-enabled run writes status='complete'. The resume filter in
    amain() interprets these per mode.
    """
    statuses = {}
    async with client.connection() as conn:
        page, _ = await conn.query(
            "MATCH (p:ProcessedDocument) RETURN p.docId AS docId, p.status AS status"
        )
        for row in page.rows:
            doc_id = _val(row, "docId")
            if doc_id:
                statuses[doc_id] = _val(row, "status", "")
    return statuses


async def _execute_batch_resilient(client: Client, statements: list) -> None:
    """Execute (stmt, params) pairs in order, riding out server stalls.

    The v0.8.5 dev server occasionally stalls for minutes while the QUIC
    client's read timeout is a hardcoded 30s. Large files spend minutes in
    their chunk batch, so they are near-guaranteed to straddle a stall
    window; replaying the whole batch would just run into the next one.
    Instead, reconnect and RESUME at the failed statement. The failure
    counter resets on any successful statement, so only consecutive
    failures exhaust the budget. QueryError is semantic and never retried.
    """
    i = 0
    fails = 0
    n = len(statements)
    while i < n:
        try:
            async with client.connection() as conn:
                while i < n:
                    stmt, params = statements[i]
                    await conn.execute(stmt, params)
                    fails = 0
                    i += 1
        except QueryError:
            raise
        except Exception as e:  # TransportError, ConnectionError, ...
            fails += 1
            if fails >= DB_MAX_RETRIES:
                raise
            log(
                f"WARN db transport error at statement {i + 1}/{n} "
                f"(attempt {fails}/{DB_MAX_RETRIES}): {str(e)[:200]}; reconnecting"
            )
            await asyncio.sleep(min(2 ** fails, DB_BACKOFF_CAP))


async def store_chunks(client: Client, doc_id: str, chunks: list, embeddings: list) -> None:
    statements = [
        (
            "MERGE (c:DocumentChunk {docId: $docId, seq: $seq}) "
            "SET c.text = $text, "
            "c.source = $source, "
            f"c.{EMB_PROP} = vectorf32(toString($vec))",
            {
                "docId": doc_id,
                "seq": seq,
                "text": text,
                "source": doc_id,
                "vec": _vec_param(emb),
            },
        )
        for seq, (text, emb) in enumerate(zip(chunks, embeddings))
    ]
    await _execute_batch_resilient(client, statements)


async def store_triples(client: Client, triples: list) -> int:
    """MERGE Entity nodes first, then MATCH+MERGE each RELATIONSHIP edge.

    Never uses chained MERGE for edges (geode v0.8.5 workaround).
    Returns the number of edges written.
    """
    triples = list(dict.fromkeys(triples))  # dedupe, keep order
    entities = set()
    for s, _, o in triples:
        entities.add(s)
        entities.add(o)
    statements = [
        ("MERGE (e:Entity {name: $name})", {"name": name}) for name in sorted(entities)
    ]
    statements += [
        (
            "MATCH (s:Entity {name: $s}) "
            "MATCH (o:Entity {name: $o}) "
            "MERGE (s)-[:RELATIONSHIP {type: $p}]->(o)",
            {"s": s, "p": p, "o": o},
        )
        for s, p, o in triples
    ]
    await _execute_batch_resilient(client, statements)
    return len(triples)


async def store_marker(
    client: Client, doc_id: str, chunk_count: int, triple_count: int, status: str
) -> None:
    await _execute_batch_resilient(client, [
        (
            "MERGE (p:ProcessedDocument {docId: $docId}) "
            "SET p.chunkCount = $chunkCount, "
            "p.tripleCount = $tripleCount, "
            "p.completedAt = $completedAt, "
            "p.status = $status",
            {
                "docId": doc_id,
                "chunkCount": chunk_count,
                "tripleCount": triple_count,
                "completedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "status": status,
            },
        )
    ])


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class Stats:
    files_done: int = 0
    files_failed: int = 0
    chunks: int = 0
    entities: set = field(default_factory=set)
    edges: int = 0
    triple_failures: int = 0
    embed_seconds: float = 0.0
    llm_seconds: float = 0.0
    db_seconds: float = 0.0


class Ingestor:
    def __init__(self, args, client: Client, embedder: Embedder, extractor,
                 embedded_done: set = None):
        self.args = args
        self.client = client
        self.embedder = embedder
        self.extractor = extractor
        # docIds whose chunks+embeddings are already stored (status='embedded');
        # a triples-enabled resume jumps straight to extraction for these.
        self.embedded_done = embedded_done or set()
        self.embed_sem = asyncio.Semaphore(args.embed_concurrency)
        self.llm_sem = asyncio.Semaphore(args.concurrency)
        self.file_sem = asyncio.Semaphore(args.file_concurrency)
        # All geode writes are serialized: with file_concurrency=4 the dev
        # server stalled >300s mid-write under concurrent vector-index
        # inserts, tripping the client's transport read timeout (observed
        # live). Embedding dominates wall time, so this costs nothing.
        self.db_sem = asyncio.Semaphore(1)
        self.stats = Stats()

    async def _db_call(self, fn, *args):
        """Run a store_* coroutine serialized (single writer) and timed.

        Transport-stall retries live inside _execute_batch_resilient(),
        which every store_* function routes through.
        """
        async with self.db_sem:
            t0 = time.monotonic()
            try:
                return await fn(*args)
            finally:
                self.stats.db_seconds += time.monotonic() - t0

    async def _embed_all(self, chunks: list) -> list:
        batches = [
            chunks[i : i + EMBED_BATCH_SIZE] for i in range(0, len(chunks), EMBED_BATCH_SIZE)
        ]

        async def one(batch):
            async with self.embed_sem:
                t0 = time.monotonic()
                result = await asyncio.to_thread(self.embedder.embed_batch, batch)
                self.stats.embed_seconds += time.monotonic() - t0
                return result

        results = await asyncio.gather(*(one(b) for b in batches))
        return [emb for batch in results for emb in batch]

    async def _extract_all(self, rel_path: str, chunks: list) -> list:
        async def one(seq, chunk):
            async with self.llm_sem:
                t0 = time.monotonic()
                try:
                    return await asyncio.to_thread(self.extractor.extract, chunk)
                except Exception as e:  # parse failure after retry, or HTTP
                    self.stats.triple_failures += 1
                    record_failure(rel_path, f"triples[chunk {seq}]", str(e))
                    return []
                finally:
                    self.stats.llm_seconds += time.monotonic() - t0

        results = await asyncio.gather(*(one(i, c) for i, c in enumerate(chunks)))
        return [t for triples in results for t in triples]

    async def process_file(self, path: Path, rel_path: str) -> None:
        async with self.file_sem:
            t_start = time.monotonic()
            try:
                text = await asyncio.to_thread(
                    path.read_text, encoding="utf-8", errors="ignore"
                )
            except OSError as e:
                self.stats.files_failed += 1
                record_failure(rel_path, "read", str(e))
                return

            chunks = chunk_text(text, self.args.chunk_size, self.args.chunk_overlap)
            del text
            if not chunks:
                record_failure(rel_path, "chunk", "no non-empty chunks produced")
                self.stats.files_failed += 1
                return

            if rel_path in self.embedded_done:
                # status='embedded' marker: chunks+vectors already stored by a
                # prior embeddings-only run — jump straight to extraction.
                log(f"RESUME file={rel_path} chunks already embedded; extracting triples only")
            else:
                try:
                    embeddings = await self._embed_all(chunks)
                except Exception as e:
                    self.stats.files_failed += 1
                    record_failure(rel_path, "embed", str(e))
                    return

                try:
                    await self._db_call(store_chunks, self.client, rel_path, chunks, embeddings)
                except Exception as e:
                    self.stats.files_failed += 1
                    record_failure(rel_path, "store_chunks", str(e))
                    return
                del embeddings
                # Chunk storage is durable from this point: count it now so
                # the DONE summary doesn't undercount when a later stage
                # (extraction/marker) fails.
                self.stats.chunks += len(chunks)
                log(f"STORED file={rel_path} chunks={len(chunks)} (marker pending)")

            triple_count = 0
            edge_count = 0
            if not self.args.skip_triples:
                triples = await self._extract_all(rel_path, chunks)
                triple_count = len(triples)
                if triples:
                    try:
                        edge_count = await self._db_call(store_triples, self.client, triples)
                        for s, _, o in triples:
                            self.stats.entities.add(s)
                            self.stats.entities.add(o)
                    except Exception as e:
                        self.stats.files_failed += 1
                        record_failure(rel_path, "store_triples", str(e))
                        await self._mark_safe(rel_path, len(chunks), 0, "failed")
                        return

            # Mode-aware marker (task-9 report): a triples-enabled run marks
            # 'complete'; an embeddings-only run marks 'embedded' so a later
            # triples pass can resume extraction without re-embedding.
            status = "complete" if not self.args.skip_triples else "embedded"
            try:
                await self._db_call(
                    store_marker, self.client, rel_path, len(chunks), triple_count, status
                )
            except Exception as e:
                self.stats.files_failed += 1
                record_failure(rel_path, "store_marker", str(e))
                return

            self.stats.files_done += 1
            self.stats.edges += edge_count
            log(
                f"OK file={rel_path} chunks={len(chunks)} triples={triple_count} "
                f"edges={edge_count} elapsed={time.monotonic() - t_start:.1f}s"
            )

    async def _mark_safe(self, rel_path, chunk_count, triple_count, status):
        try:
            await self._db_call(
                store_marker, self.client, rel_path, chunk_count, triple_count, status
            )
        except Exception as e:
            record_failure(rel_path, "store_marker", str(e))


# ---------------------------------------------------------------------------
# Validation search (--search)
# ---------------------------------------------------------------------------


async def run_search(client: Client, embedder: Embedder, query_text: str, k: int) -> None:
    """knn search over DocumentChunk with the query embedding shrunk by the
    exact same shrink_embedding() path as ingestion (ENG-3911)."""
    emb = (await asyncio.to_thread(embedder.embed_batch, [query_text]))[0]
    params = {"vec": _vec_param(emb), "k": int(k)}
    knn_query = (
        "MATCH (d:DocumentChunk) "
        f"WHERE id(d) IN knn('{EMB_PROP}', vectorf32(toString($vec)), $k, 'cosine') "
        f"WITH d, vector_cosine(d.{EMB_PROP}, vectorf32(toString($vec))) AS score "
        "RETURN d.docId AS docId, toString(d.seq) AS seq, d.text AS text, "
        "toString(score) AS score"
    )
    # Fallback: knn() can hang while chunk_emb_idx absorbs concurrent inserts
    # (observed live on v0.8.5). A full scan is O(corpus) but bounded; fetch a
    # wide window and let the client-side sort pick the real top-k.
    scan_params = dict(params, k=int(k) * 50)
    scan_query = (
        f"MATCH (d:DocumentChunk) WHERE d.{EMB_PROP} IS NOT NULL "
        f"WITH d, vector_cosine(d.{EMB_PROP}, vectorf32(toString($vec))) AS score "
        "WHERE score > 0.1 "
        "RETURN d.docId AS docId, toString(d.seq) AS seq, d.text AS text, "
        "toString(score) AS score LIMIT $k"
    )
    try:
        async with client.connection() as conn:
            page, _ = await asyncio.wait_for(conn.query(knn_query, params), timeout=60)
    except (QueryError, asyncio.TimeoutError, Exception) as e:
        log(f"WARN knn fast path unavailable ({e!r:.150}); full-scan fallback")
        async with client.connection() as conn:
            page, _ = await conn.query(scan_query, scan_params)
    hits = []
    for row in page.rows:
        try:
            score = float(_val(row, "score", "0.0") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        hits.append((score, _val(row, "docId", "?"), _val(row, "seq", "?"),
                     str(_val(row, "text", ""))))
    # Client-side sort is the ranking authority: float projections decode as
    # NULL through GQL ORDER BY in v0.8.5.
    hits.sort(key=lambda h: h[0], reverse=True)
    hits = hits[: int(k)]
    print(f"query: {query_text!r} -> {len(hits)} hits")
    for score, doc_id, seq, text in hits:
        snippet = re.sub(r"\s+", " ", text)[:160]
        print(f"  score={score:.4f} doc={doc_id} seq={seq} text={snippet}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Headless KB -> geode KG ingestion driver")
    p.add_argument("--dir", help="Directory of .md files to ingest (required unless --search)")
    p.add_argument(
        "--search",
        help="Skip ingestion; run a validation knn search for this query text",
    )
    p.add_argument("--search-k", type=int, default=5, help="knn k for --search (default 5)")
    p.add_argument(
        "--server",
        default="127.0.0.1:3141",
        help="geode server host:port (default 127.0.0.1:3141)",
    )
    p.add_argument("--user", default=os.getenv("GEODE_USERNAME", "geode"))
    p.add_argument("--password", default=os.getenv("GEODE_PASSWORD", "geodesic"))
    p.add_argument("--limit", type=int, default=0, help="Process at most N files (0 = all)")
    p.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Concurrent LLM chunk-extraction requests (default 8)",
    )
    p.add_argument(
        "--embed-concurrency",
        type=int,
        default=4,
        help="Concurrent embedding batch requests (default 4)",
    )
    p.add_argument(
        "--file-concurrency",
        type=int,
        default=4,
        help="Files processed in flight simultaneously (default 4)",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Chunk size in characters (default {CHUNK_SIZE})",
    )
    p.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help=f"Chunk overlap in characters (default {CHUNK_OVERLAP})",
    )
    p.add_argument(
        "--skip-triples",
        action="store_true",
        help="Skip triple extraction (embeddings + chunks only); marker status='embedded'",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Mode-aware resume from ProcessedDocument markers: a triples-enabled "
            "run skips status='complete' files and jumps straight to extraction "
            "for status='embedded' ones; a --skip-triples run skips both"
        ),
    )
    return p.parse_args(argv)


async def amain(args) -> int:
    host, _, port = args.server.partition(":")
    port = int(port or "3141")

    if args.search:
        client = Client(
            host=host,
            port=port,
            username=args.user,
            password=args.password,
            skip_verify=True,
            transport_type=TransportType.QUIC,
        )
        embedder = Embedder(
            os.getenv("EMBEDDINGS_API_URL", DEFAULT_EMBEDDINGS_URL),
            DEFAULT_EMBEDDINGS_MODEL,
        )
        await run_search(client, embedder, args.search, args.search_k)
        return 0

    if not args.dir:
        print("error: --dir is required unless --search is given", file=sys.stderr)
        return 2
    root = Path(args.dir).resolve()
    if not root.is_dir():
        print(f"error: --dir {root} is not a directory", file=sys.stderr)
        return 2

    extractor = None
    if not args.skip_triples:
        api_key = os.getenv("VLLM_API_KEY", "")
        if not api_key:
            print(
                "error: VLLM_API_KEY is not set (required unless --skip-triples)",
                file=sys.stderr,
            )
            return 2
        extractor = TripleExtractor(
            os.getenv("VLLM_API_URL", DEFAULT_LLM_URL), DEFAULT_LLM_MODEL, api_key
        )

    embedder = Embedder(
        os.getenv("EMBEDDINGS_API_URL", DEFAULT_EMBEDDINGS_URL),
        DEFAULT_EMBEDDINGS_MODEL,
        truncate_chars=args.chunk_size + args.chunk_overlap,
    )

    client = Client(
        host=host,
        port=port,
        username=args.user,
        password=args.password,
        skip_verify=True,  # self-signed dev cert (GEODE_ALLOW_INSECURE_TLS)
        transport_type=TransportType.QUIC,
    )

    try:
        await preflight_vector_write(client)
    except QueryError as e:
        print(
            f"error: preflight vectorf32 write of {EMBED_DIM} dims failed "
            f"(ENG-3911 regression?): {e}",
            file=sys.stderr,
        )
        return 2

    files = sorted(p for p in root.rglob("*.md") if p.is_file())
    total_found = len(files)

    skipped_resume = 0
    embedded_done = set()
    if args.resume:
        statuses = await fetch_marker_statuses(client)
        if args.skip_triples:
            # embeddings-only: both 'complete' and 'embedded' already have
            # their chunks+vectors stored — nothing left to do for either.
            done = {d for d, st in statuses.items() if st in ("complete", "embedded")}
        else:
            # triples-enabled: only 'complete' is fully done; 'embedded' files
            # stay selected but skip re-embedding/re-storing chunks.
            done = {d for d, st in statuses.items() if st == "complete"}
            embedded_done = {d for d, st in statuses.items() if st == "embedded"}
        before = len(files)
        files = [p for p in files if str(p.relative_to(root)) not in done]
        skipped_resume = before - len(files)

    if args.limit > 0:
        files = files[: args.limit]

    log(
        f"START dir={root} server={args.server} found={total_found} "
        f"resume_skipped={skipped_resume} embedded_resume={len(embedded_done)} "
        f"selected={len(files)} chunk(size={args.chunk_size} overlap={args.chunk_overlap}) "
        f"concurrency(llm={args.concurrency} embed={args.embed_concurrency} "
        f"files={args.file_concurrency}) skip_triples={args.skip_triples}"
    )

    ing = Ingestor(args, client, embedder, extractor, embedded_done)
    t0 = time.monotonic()
    results = await asyncio.gather(
        *(ing.process_file(p, str(p.relative_to(root))) for p in files),
        return_exceptions=True,
    )
    for path, res in zip(files, results):
        if isinstance(res, BaseException):
            ing.stats.files_failed += 1
            record_failure(str(path.relative_to(root)), "unexpected", repr(res))
    wall = time.monotonic() - t0

    s = ing.stats
    minutes = wall / 60 if wall > 0 else 1e-9
    stage_totals = {
        "embed": s.embed_seconds,
        "llm": s.llm_seconds,
        "db": s.db_seconds,
    }
    bottleneck = max(stage_totals, key=stage_totals.get) if any(stage_totals.values()) else "n/a"
    log(
        f"DONE files_ok={s.files_done} files_failed={s.files_failed} "
        f"chunks={s.chunks} unique_entities={len(s.entities)} edges={s.edges} "
        f"triple_chunk_failures={s.triple_failures} wall={wall:.1f}s "
        f"throughput={s.files_done / minutes:.2f} files/min, "
        f"{s.chunks / minutes:.1f} chunks/min; "
        f"stage seconds (concurrent-sum): embed={s.embed_seconds:.1f} "
        f"llm={s.llm_seconds:.1f} db={s.db_seconds:.1f} -> bottleneck={bottleneck}"
    )
    return 0 if s.files_failed == 0 else 1


def main():
    args = parse_args()
    lock_held = False
    if not args.search:  # --search is read-only; no write-overlap hazard
        acquire_lock()
        lock_held = True
    try:
        sys.exit(asyncio.run(amain(args)))
    except KeyboardInterrupt:
        log("INTERRUPTED by user")
        sys.exit(130)
    finally:
        if lock_held:
            release_lock()


if __name__ == "__main__":
    main()
