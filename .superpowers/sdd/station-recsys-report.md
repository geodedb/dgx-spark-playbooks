# station-rec-sys: Geode vector retrieval backend — integration report

Date: 2026-08-27
Branch: `feat/geode-085-upgrade`
Scope: make Geode an optional FAISS-alternative retrieval backend for the
HLLM two-tower recommender playbook (`nvidia/station-rec-sys`).

## What was built

- **`assets/geode_retriever.py`** (new): Geode-backed inner-product top-K
  retrieval, FAISS-equivalent API.
  - Loads `hllm_item_embeddings.npy` + `hllm_item_id_map.npy` into Geode as
    `RecItem {itemId, title, hllmIdx, emb: vectorf32}` nodes — idempotent
    (MERGE on `itemId`; a count fast-path skips the load entirely on
    re-launch when all rows are present).
  - Embeddings stored as true VectorF32 via param-bound
    `vectorf32(toString($vec))` (the proven pattern from
    `nvidia/multi-agent-chatbot/assets/backend/geode_vector_store.py`).
  - HNSW index: `CREATE INDEX rec_emb_idx ON RecItem(emb) USING vector WITH
    (metric: 'dot')` — **`'dot'` is accepted** by the engine
    (`src/eval.zig:2148` also accepts `dot_product` / `inner_product`;
    `knn()` and `vector_dot()` accept `'dot'`).
  - `top_k(user_vec, k)` / FAISS-shaped `search(query, k) -> (D, I)` via
    `knn('emb', vectorf32(toString($vec)), $k, 'dot')` for candidate
    selection + `vector_cosine(...)` for the score projection (== inner
    product for HLLM's L2-normalized embeddings; `vector_dot` cannot be
    used — see NEW bug below), `ORDER BY score DESC`, with a full-scan
    fallback on `QueryError` (index not yet built).
  - Resilience: 300s client read timeout + reconnect-and-retry per item in
    the load path (see ENG-3914 below), b-tree index on `itemId` so MERGE
    lookups don't O(n)-scan, and — critically — the **HNSW index is created
    AFTER the bulk load** (see ENG-3917 below).
- **`assets/app.py`**: backend switch `RETRIEVER_BACKEND=faiss|geode`
  (default `faiss`, upstream behavior preserved). Geode connection via
  `GEODE_URL` / `GEODE_USERNAME` / `GEODE_PASSWORD`. Single dispatch point
  `retrieve_top_k()` replaces the direct `faiss_index.search` call; the
  LightGBM re-ranker, PPO pricing, and UI are untouched.
- **`assets/setup.sh`**: installs
  `geode-client @ git+https://gitlab.com/devnw/codepros/geode/geode-client-python.git@v0.5.0`
  into the playbook venv (non-fatal if it fails; only needed for the geode
  backend). Verified the URL resolves and installs.
- **`README.md`**: new "Optional: Geode vector retrieval backend" section
  (requirements incl. geode:dev, launch example, expected output).

## Dimension note

HLLM item embeddings are **2048-dim** (`extract_embeddings.py` →
`(16461, 2048)`), which exceeds the 1792-dim VectorF32 cap of geode
v0.8.5 (ENG-3911). The playbook therefore requires the **geode:dev**
image (cap 65535), consistent with repo policy; this is documented in the
README and module docstring.

## Verification (synthetic end-to-end)

Isolated `geode:dev` container (QUIC, 127.0.0.1:3173), throwaway uv venv
with geode-client v0.5.0 + numpy. 1000 synthetic items at the real dim
(2048, unit-normalized, `[PAD]` row 0 mirroring the real id-map layout),
loaded through `geode_retriever.load_items`, then `top_k` (k=100) for 5
random query vectors checked against numpy brute-force inner-product
top-100 (ties at the k-th score allowed; score parity <= 1e-3; descending
order verified).

RESULT: **PASS** — all 5 queries returned the EXACT brute-force top-100 set
(recall 1.000, no tie slack needed), correctly sorted, with scores matching
numpy inner products within 1e-6. The idempotent re-launch fast path
verified (second `load_items` call: 0.17 s, no rewrites). The load itself
survived one ENG-3914 stall window (>300 s at item 667) via the built-in
reconnect-retry. knn's HNSW 'dot' ordering was exact at n=1000, k=100.

## Engine issues encountered (all reproduced, deduped against GitLab issues and ENG-3903..3923)

1. **Full HNSW rebuild per write (known: ENG-3917, triaged via ENG-3914/15).**
   With the vector index in place, every insert triggers a full drop+rebuild:
   measured per-insert latency at dim 2048 growing 35 ms (n<=35) → 0.35 s
   (n=40) → 0.74 s (n=50) → 5.9 s (n=59) → >30 s (n~62, client timeout);
   same shape at dim 128/512/1024. This makes indexed bulk loads O(N²·dim).
   Mitigation adopted (and already recommended in ENG-3917): bulk-load
   first, `CREATE INDEX` once afterwards.
2. **Multi-minute write stalls under sustained vector ingest (known:
   ENG-3914, In Progress).** On a fresh server, bulk unindexed loads of
   2048-dim vectors stalled a write for >300 s at item ~512 (run 1) and
   ~840 (run 2); the storage layer logs
   `bounded checkpoint skipped: rwlock busy past deadline` every ~35 s with
   `rwlock_state=4294967296` (one leaked/stuck shared hold), and the server
   burns 1.5–35 CPU cores. The retriever's reconnect-retry + 300 s read
   timeout is the client-side mitigation (mirrors ENG-3914's guidance).
3. **MERGE-by-key without a property index scans the label per statement**,
   which is quadratic on re-loads over vector-heavy nodes — mitigated with
   a b-tree index on `itemId` before the MERGE loop.

**NEW bug filed — GitLab devnw/codepros/geode/geode#873:**
`vector_dot()` truncates its result to an integer
(`src/eval_extended.zig` `vectorDot` returns `.Int = @intFromFloat(sum)`),
so unit-vector dot products silently collapse to 0 — score projections and
`ORDER BY score` over `vector_dot` are meaningless while the knn top-k set
still looks correct. `vector_cosine`/`vector_l2` return scale-6 Decimals
correctly. Deduped against ENG-3903..3923 (ENG-3909 is client-side FLOAT
NULL-decoding — different) and all open GitLab issues.
The retriever works around it by projecting `vector_cosine`, which equals
the inner product for HLLM's normalized embeddings.

Also posted a supplementary evidence comment on ENG-3914 (stalls reproduce
with NO index present on geode:dev; checkpoint-starvation log signature
`rwlock_state=2^32`; post-stall CPU burn; v0.5.0 `read_timeout` +
reconnect-retry rides it out).

## Caveats / follow-ups

- Loading the real 16,460-item catalog will take roughly 5–15 min
  (unindexed inserts ~10–30 ms each) plus the one-time `CREATE INDEX`
  backfill, and may hit one or more ENG-3914 stall windows (the retry
  logic rides them out). Once loaded, re-launches skip straight to serving.
- Incremental catalog updates with the index present hit ENG-3917's
  rebuild-per-write; until that lands, prefer full reloads into a fresh
  data-dir or drop/re-create the index around updates.
- `RETRIEVER_BACKEND=faiss` remains the default; upstream behavior is
  byte-for-byte unchanged on that path.
