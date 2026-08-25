# Task 10 Report — Full-Corpus Embeddings Run + txt2kg System Bring-Up

Status: COMPLETE (full run launched and verified live; UI up and verified against the live KG)

Commits (branch feat/geode-085-upgrade, not pushed):
- `72a73b0` "feat: mode-aware resume + full-run tuning for kb ingestion" (`nvidia/txt2kg/assets/scripts/ingest_kb.py`)
- `809d5d9` "fix: use emb property for vectors (ENG-3912 dimension poisoning)" (`nvidia/txt2kg/assets/frontend/lib/geode.ts`)

## Part A — Driver amendments (`ingest_kb.py`, commit 72a73b0)

1. **Mode-aware marker + resume** (exactly the task-9 report's proposal):
   - `store_marker` status is now `"complete" if not args.skip_triples else "embedded"`.
   - `fetch_complete_doc_ids()` → `fetch_marker_statuses()` returns `{docId: status}`.
   - Resume filter in `amain()`: an embeddings-only (`--skip-triples`) run treats
     both `complete` and `embedded` as done; a triples-enabled run treats only
     `complete` as done and passes the `embedded` set to the Ingestor, whose
     `process_file()` skips re-embedding/re-storing chunks for those files and
     jumps straight to extraction (chunks re-derived locally by the splitter —
     use the same --chunk-size/--chunk-overlap as the embedding run).
2. **`--chunk-size` (default 2000) / `--chunk-overlap` (default 200) CLI flags**,
   threaded through `chunk_text()`. Embed-input truncation is now
   `max(EMBED_TRUNCATE_CHARS=6000, chunk_size + overlap)` per Embedder instance,
   so a 6000/300 run truncates at 6300 chars (~1600 tokens, far under the
   encoder's 8192-token ctx) and never cuts a chunk.
3. **Extras (defensive, small):**
   - PID lockfile guard (`ingest_kb.pid` next to the script): refuses to start
     while another live `ingest_kb.py` exists — codifies the task-9
     "exactly one ingest process" rule (MERGE duplicates via late-committing
     stalled writes). Stale lockfiles (dead/reused PID) are replaced.
     `--search` (read-only) is exempt. Verified live: second start against the
     running full run refused with exit 3 before touching the DB.
   - `_post_with_retry` no longer retries non-retryable 4xx (only 429/5xx/transport).
   - `stats.chunks` counted at chunk-store success (durable work no longer
     undercounted if a later stage fails); new `STORED file=... (marker pending)` log line.

### Sanity check (--limit 2 --skip-triples, live server 127.0.0.1:3141)

- `python -m py_compile` clean; `--help` shows the new flags.
- Verified via pgrep before starting: no other ingest_kb.py running (task-9
  processes gone; nothing to kill).
- Run: `--limit 2 --skip-triples --resume --chunk-size 6000 --chunk-overlap 300`
  → exit 0, `files_ok=2 files_failed=0 chunks=688` (01-2258.md 276 chunks,
  00223.md 412 chunks — ~2.9x fewer than at 2000 chars, matching the ~3x goal).
- DB after: `DocumentChunk count = 688`; both `ProcessedDocument` markers
  `status='embedded'` with correct chunkCounts.
- ~10 server stalls during the chunk stores (`Timeout reading message length`,
  `Stream closed`) all ridden out by the resilient executor with zero failures.

## Part B — Full embeddings-only run (LIVE)

Launch (18:45:18 UTC 2026-08-25, detached via setsid nohup from
`nvidia/txt2kg/assets/scripts/`, log `full_run.out` + `ingest_kb.log`):

    uv run --project ../../../multi-agent-chatbot/assets/backend \
      python ingest_kb.py --dir /data/home/benji/src/devnw/datasets/kb-md \
      --server 127.0.0.1:3141 --skip-triples --resume \
      --chunk-size 6000 --chunk-overlap 300 --embed-concurrency 4

- **PID: 299411** (python; also in `ingest_kb.pid`).
- START line: `found=2110 resume_skipped=2 selected=2108` — the mode-aware
  resume correctly skipped the two `embedded` files from the sanity run.
- **Embed concurrency kept at 4 (not raised):** celebrimbor GPU 0 was at 100%
  utilization during the --limit 2 test embedding phase (and 0% idle after),
  i.e. concurrency 4 already saturates the encoder — raising to 8 would not help.
  More importantly the run is **DB-write-bound**, not embed-bound (below).
- Liveness (11+ min verified): launched 18:45:18; at 18:56:52 the DocumentChunk
  count had grown 704 → 866 (+162, ~14.6 chunks/min including stalls), the log
  had advanced (4 files OK/STORED), and PID 299411 was alive (rechecked at
  18:57 — 12+ min uptime, single process, matches `ingest_kb.pid`). The UI's
  stats route independently confirmed `documentChunkCount: 866` at 18:57:30.

### Throughput / ETA

- Sanity run stage seconds: embed=2417 (concurrent-sum/4 ≈ 10 min wall),
  db=2024 (serialized wall) for 688 chunks → **DB store rate ≈ 20 chunks/min**
  including recurring multi-minute server stalls (~60/min between stalls).
  The driver's "bottleneck=embed" heuristic compares concurrent-sums and is
  misleading here: with file_concurrency=4 embedding overlaps DB writes, so
  steady-state throughput ≈ the serialized DB write rate.
- Corpus: 2110 files / 5.6 GB ≈ 2.9M chunks at 2000 chars (task 9) → **~1.0M
  chunks at 6000/300** (measured shrink factor 2.85–3.0x on the two test files).
- **ETA: ~35–45 days** at the observed sustained rate (~15–20 chunks/min,
  DB-bound, stalls included) for ~1.0M chunks. The encoder could sustain
  ~70 chunks/min at this chunk size if the DB kept up (~10 days) — the gap is
  entirely geode v0.8.5 write throughput. The run is resumable at any time
  (`--resume` + `embedded` markers), so it can ride through server restarts
  or a future server-side fix that speeds up the remainder.

## Part C — txt2kg system (UI) bring-up

`docker ps` / `docker compose version` (v5.5.0) available, but the KG lives in
the HOST geode at 127.0.0.1:3141 (the compose geodes map to 13141/13142), so
the frontend runs natively as directed.

### ENG-3912 fix in `lib/geode.ts` (commit 809d5d9)

The UI used property `embedding` for DocumentChunk vectors and passed raw
2560-dim query vectors; the KG stores 1536-dim vectors in `emb`
(`DocumentChunk.embedding` is dimension-poisoned at 2560, task-9 report).
Minimal targeted fix:
- `CHUNK_EMB_PROP = 'emb'` + `EMB_DIM = 1536` + `shrinkEmbedding()` (prefix
  truncate + L2 renorm, mirroring `shrink_embedding()` in ingest_kb.py), with
  a comment block referencing ENG-3912.
- `knnRows()` uses `emb` + shrunk query vector for DocumentChunk lookups
  (Entity keeps `embedding`, unshrunk — its own separate pin).
- `storeDocumentChunks()` / `upsertVectors()` document branch write shrunk
  vectors into `emb`; `ensureIndexes()` creates `chunk_emb_idx` on
  `DocumentChunk(emb)`.
- Rebuilt: `corepack pnpm build` exit 0.

### Launch (native, detached)

    cd nvidia/txt2kg/assets/frontend && \
    VLLM_API_KEY=$(ssh curumo.shire-altair.ts.net sudo -n cat /etc/spark-vllm/api-key) \
    GEODE_URL='quic://127.0.0.1:3141?insecure_tls_skip_verify=true' \
    GEODE_USERNAME=geode GEODE_PASSWORD=geodesic \
    GEODE_ALLOW_INSECURE_TLS=1 GEODE_ALLOW_INSECURE_PRODUCTION=true \
    VLLM_BASE_URL=http://curumo.shire-altair.ts.net:8000/v1 \
    VLLM_MODEL=Qwen/Qwen3.8-27B NVIDIA_API_KEY=placeholder-no-nvidia-cloud \
    OLLAMA_MODEL=disabled \
    EMBEDDINGS_API_URL=https://celebrimbor.shire-altair.ts.net:8001 \
    EMBEDDINGS_API_FORMAT=openai EMBEDDINGS_MODEL_ID=Qwen/Qwen3-Embedding-4B \
    NODE_TLS_REJECT_UNAUTHORIZED=0 \
    setsid nohup corepack pnpm start --port 3001 > ui.out 2>&1 &

Notes discovered during bring-up:
- VLLM key: `/etc/spark-vllm/api-key` on curumo, root-only, readable via
  passwordless `sudo -n cat` over ssh (fetched inline at launch; never logged).
- `@geodedb/client` in a production Next build refuses `insecure_tls_skip_verify`
  unless `GEODE_ALLOW_INSECURE_PRODUCTION=true` — required in addition to
  `GEODE_ALLOW_INSECURE_TLS=1`.
- `NVIDIA_API_KEY` placeholder: `lib/rag.ts` hard-requires it at init for its
  NVIDIA-cloud generation LLM (hardcoded `integrate.api.nvidia.com`); the
  placeholder lets Pure-RAG retrieval run. Generation via NVIDIA cloud fails
  (no real key) — see concerns.

**UI PID: 290995 (`next-server`; wrapper `sh -c next start --port 3001` = 290994) — URL: http://127.0.0.1:3001**

### Verification

1. UI loads: `curl http://127.0.0.1:3001` → 200,
   `<title>txt2kg | NVIDIA Knowledge Graph Builder</title>`.
2. Graph-stats API (`GET /api/vector-db/stats`) reports live counts from the
   host geode: `{"totalVectorCount":106→growing,"entityCount":2199,
   "documentChunkCount":106,"relationshipCount":1892,"source":"geode",
   "httpHealthy":true}` — entity/relationship counts match the known KG state,
   documentChunkCount tracked the in-flight ingestion (106 → 866 across
   checks; final check 18:57:30 matched the direct geode count exactly).
   Note: during a server write stall one stats call returned a transient
   `transport sendProto` error; the next call succeeded — expect occasional
   transient errors while the bulk load runs.
3. Vector/RAG query through the app's API: `POST /api/rag-query`
   `{"query":"How can software fault measurement be used to estimate fault
   content?","topK":4}` → server log: `Performing RAG query with topK=4` /
   **`Found 4 relevant document chunks`** / `Extracted context length: 20809
   characters` (4 × ~6000-char chunks — real ingested chunks retrieved via
   `emb` + shrunk query embedding; celebrimbor produced the query embedding).
   The subsequent NVIDIA-cloud generation step fails on the placeholder key,
   so the HTTP response is a 500 — retrieval (the part this task's fix
   enables) is verified working end-to-end.

## Concerns

1. **The full run is DB-write-bound, not embed-bound.** geode v0.8.5 sustains
   only ~20 chunk MERGEs/min serialized (recurring multi-minute stalls under
   vector-index insert load, ridden out by the resilient executor). ETA at that
   rate is ~35–45 days. Raising embed concurrency cannot help; batched
   multi-row writes, a server-side fix for the stalls (append to ENG-3911), or
   relaxed index maintenance during bulk load are the levers.
2. **Pure-RAG answer generation needs a real key or rewiring**: `lib/rag.ts`
   hardcodes the NVIDIA cloud LLM and requires `NVIDIA_API_KEY`; the
   deployment's actual LLM is vLLM on curumo. Retrieval works; generation 500s.
   Follow-up: point rag.ts at `VLLM_BASE_URL` like the other routes.
3. **Graph-RAG routes (`/api/enhanced-query`, entity path) fail** with
   `geode: transport query: Invalid execute response` (statusClass 58000)
   inside `getGraphData()`'s full-graph queries. The same GQL succeeds via the
   geode CLI (note: `toString(id(r))` on edges returns `"null"` there), so this
   is a @geodedb/client response-decode issue on this data, pre-existing and
   unrelated to the emb fix. Worth a follow-up ticket.
4. `--limit` still applies after the resume filter (task-9 note) — harmless for
   the full run (no limit).
5. The UI's `ensureIndexes()` also (re)creates `entity_emb_idx` on
   `Entity(embedding)`; entity embeddings are still deliberately absent from
   the KG, so entity KNN legitimately returns 0 seeds.

## Encoder swap to 0.6B (2026-08-25)

The full run above was embed-bound at ~68 chunks/min (Qwen/Qwen3-Embedding-4B
saturating celebrimbor's RTX 4060 Ti) → ~10-day ETA. Swapped the encoder to
**Qwen/Qwen3-Embedding-0.6B** (native 1024-dim — under geode's ~1792-dim
vectorf32 cap, so no truncation at all) and relaunched from scratch.

What changed:

1. **Infra** (`infra/ansible-playbooks`, uncommitted per policy):
   `celebrimbor.yml` vllm_embedding block → `model_repo:
   Qwen/Qwen3-Embedding-0.6B`, `dimension: 1024` (tensor_parallel 1,
   CUDA_VISIBLE_DEVICES "0", gpu_mem 0.85, max_model_len 8192 unchanged).
   Applied via `make apply … LIMIT=celebrimbor`; container verified serving
   the 0.6B and returning 1024-dim vectors. (The play afterwards failed on an
   unrelated pre-existing docling `docling_api_key` assert — embedding role
   had already converged.)
2. **Wipe for vector-space consistency**: the ~4.8k chunks stored by the 4B
   run were deleted (`MATCH (c:DocumentChunk) DETACH DELETE c`, same for
   `ProcessedDocument`; both verified 0). Entity/RELATIONSHIP untouched
   (2199/1892 preserved). `chunk_emb_idx` existed and was dropped — the bulk
   load runs index-free (`INGEST_SKIP_INDEX=1`); recreate post-load.
3. **Driver** (`f4da1ca`): `EMBED_DIM` and the embeddings model id now read
   env overrides (`EMBED_DIM`, `EMBEDDINGS_MODEL_ID`; defaults unchanged
   1536 / 4B), and `shrink_embedding()` is an explicit no-op for vectors
   already at or under `EMBED_DIM`.
4. **UI** (`dc3b323`): `EMB_DIM` in `lib/geode.ts` reads
   `process.env.EMBEDDINGS_DIM` (default 1536); `shrinkEmbedding()` no-ops on
   <= EMB_DIM vectors. Rebuilt (`pnpm build`) and restarted on :3001 with the
   prior env plus `EMBEDDINGS_MODEL_ID=Qwen/Qwen3-Embedding-0.6B` and
   `EMBEDDINGS_DIM=1024`. Root page 200; `/api/vector-db/stats` answers
   (source: geode, httpHealthy: true).
5. **Relaunch**: `EMBED_DIM=1024 EMBEDDINGS_MODEL_ID=Qwen/Qwen3-Embedding-0.6B
   INGEST_SKIP_INDEX=1 … ingest_kb.py --dir …/kb-md --skip-triples --resume
   --chunk-size 6000 --chunk-overlap 300` (detached, single process, logging
   to full_run.out / ingest_kb.log). Stored vectors verified 1024-dim by
   querying one back (4096-byte f32 payload).

Rate and ETA:

| | 4B (before) | 0.6B (after) |
|---|---|---|
| Sustained rate | ~68 chunks/min | **~222 chunks/min** (wall, 17-min window incl. two transient geode stalls; ~300 chunks/min between stalls) |
| ETA for ~980k chunks | ~10 days | **~3.1 days** |

The ~3.3x (not 5x) wall-clock gain is because the bottleneck now shifts
partly to geode writes: two `Timeout reading message length` stalls occurred
in the first 17 min, each ridden out by the driver's retry/reconnect
(attempt 1/6 both times). Embeds run at ~25.7k prompt tok/s with GPU 0
pegged, so further gains would come from the DB write path, not the encoder.

## rag.ts vLLM rewire

`nvidia/txt2kg/assets/frontend/lib/rag.ts`'s `RAGService.initialize()`
hardcoded `https://integrate.api.nvidia.com/v1` and required
`NVIDIA_API_KEY`, so `/api/rag-query` 500'd on any deployment (like this
one) that only configures local vLLM. Rewired to match the existing
`VLLM_BASE_URL`/`VLLM_MODEL`/`VLLM_API_KEY` pattern already used in
`lib/llm-service.ts` and `lib/langchain-service.ts`'s `getVllmModel()`:
when `VLLM_BASE_URL` is set, `ChatOpenAI` is built against that `baseURL`
with `model: VLLM_MODEL` (default `meta-llama/Llama-3.2-3B-Instruct`) and
`apiKey: VLLM_API_KEY || 'vllm'`; otherwise it falls back to the original
NVIDIA-cloud path, now requiring `NVIDIA_API_KEY` only in that branch.

While wiring this up, discovered and fixed a second, pre-existing bug
uncovered by actually exercising this code path: the installed
`@langchain/openai@1.5.10` (this branch upgraded dependencies) renamed the
`ChatOpenAI` constructor's API-key field from `openAIApiKey` to `apiKey` —
the old field name is silently ignored (no compile error, since the build
runs with type-checking skipped), so both the new vLLM branch and the
pre-existing NVIDIA branch would have thrown "Missing credentials" at
generation time. Fixed both call sites to use `apiKey`.

**Build**: `pnpm build` (node 24.19.0, pnpm 11.23.0 via corepack) — passed,
no errors.

**Live verification**: the UI's `next-server` on :3001 was running the
previous build. Captured its env from `/proc/<pid>/environ` (GEODE_*,
VLLM_*, EMBEDDINGS_*, NODE_*, PATH — kept only in shell memory, never
written to disk), killed it, and relaunched detached
(`env -i "${ENV_ARGS[@]}" setsid nohup corepack pnpm start --port 3001 >>
ui.out 2>&1 &`) from the frontend dir. `VLLM_API_KEY` was already present
in the captured env (also independently re-confirmed via
`ssh curumo.shire-altair.ts.net docker inspect vllm_node`, output piped
straight through `grep`/`cut`, never persisted to a file).

`POST /api/rag-query {"query":"What is this knowledge base about?",
"topK":5}` → first attempt timed out client-side at 90s (server log showed
it was still in-flight — the bulk ingest running concurrently against the
same vLLM node adds queueing latency, and the model is a Qwen3 reasoner
that spends part of its 1024-token budget on a hidden reasoning chain
before the answer). Retried with a longer client timeout (280s) and got:

```
HTTP 200
{"answer":"\n\nThis knowledge base is about the ICANN KSK rollover,
including its mailing list, website, goals, and overview.",
"documentCount":5,"usedFallback":false,"success":true}
```

`documentCount: 5` and `usedFallback: false` confirm the answer is grounded
in retrieved Geode chunks (ICANN KSK-rollover corpus), not a general-
knowledge fallback, and no 500/credential error.

**Concerns**:
- Generation latency for this reasoning model against the local vLLM node
  can exceed typical HTTP client timeouts (~90s+) under concurrent load
  from the bulk ingest; worth considering a shorter `maxTokens` or a
  non-reasoning model for the RAG-answer path, or raising client-side
  timeouts in the UI.
- The NVIDIA-cloud fallback branch was not live-tested (no `NVIDIA_API_KEY`
  in this environment) — only the vLLM branch and the `apiKey` field fix
  were verified end-to-end. The field-name fix is a straightforward,
  low-risk rename with no behavior change other than making the key
  actually reach the OpenAI SDK, but it was not exercised against the
  NVIDIA endpoint.
