# Geode v0.8.5 Upgrade + Knowledge-Graph Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `dgx-spark-playbooks` to geode v0.8.5 + latest clients, remove obsolete bug workarounds and adopt new vector features, run the txt2kg system locally against the DGX cluster (Qwen/Qwen3.8-27B), deploy an embedding encoder on celebrimbor via ansible, ingest the 2,110-file markdown corpus from `haleth:/data/kb/md` into a geode knowledge graph, and file every geode bug found as a JIRA ENG issue linked via the GitLab Project field.

**Architecture:** The txt2kg pipeline (Next.js frontend + geode server) runs locally; LLM triple-extraction calls go to the DGX vLLM cluster over the tailnet; embedding calls go to a vLLM pooling server deployed on celebrimbor; all graph + vector storage is a local geode v0.8.5. A headless batch driver feeds the corpus through the same code path the UI uses.

**Tech Stack:** geode v0.8.5 (engine), `@geodedb/client` v1.1.0 (Node), `geode-client` v0.4.1 (Python), Next.js/TypeScript frontend, Docker Compose, Ansible (`vllm_embedding` role), vLLM (DGX Qwen3.8-27B gen @ curumo:8000, Qwen3-Embedding-4B on celebrimbor:8001), JIRA ENG + GitLab.

## Global Constraints

- Playbooks repo: `/data/home/benji/src/devnw/codepros/geode/dgx-spark-playbooks` (symlink; real path `/data/home/benji/src/gl/gitlab.com/devnw/codepros/geode/dgx-spark-playbooks`). All work on branch `feat/geode-085-upgrade`; commit per task; do NOT push to `main`.
- Infra repo `/data/home/benji/src/devnw/infra/ansible-playbooks` has **uncommitted user edits** — make minimal additive changes, do NOT commit or revert anything there.
- Geode engine: v0.8.5 (`/usr/bin/geode`). Node client v1.1.0, Python client v0.4.1 (tags in sibling repos under `/data/home/benji/src/gl/gitlab.com/devnw/codepros/geode/`).
- v0.8.5 verified facts (from spike, `scratchpad/geode-spike/`):
  - FIXED → remove workarounds: #477 (scientific notation in `vectorf32`), #476 (`vector_cosine` in `WITH`), plain-list args to vector functions, `count(DISTINCT)`.
  - STILL BROKEN: #478 leading-negative first list element (`[-0.1, …]` → parse error `42000`). Fix by **parameter binding**: pass the embedding as a param and store the list directly (`SET e.embedding = $vec`). `vectorf32($vec)` is invalid — store `$vec` as-is; vector functions accept LIST properties.
  - NEW BUG: chained `MERGE (a)…MERGE (b)…MERGE (a)-[:R]->(b)` **silently creates no relationship** (status 00000). Use separate `MATCH …, … MERGE (a)-[:R]->(b)` statement (verified idempotent).
  - `knn('prop', <vec>, k, 'metric')` returns node-id list; use `MATCH (t:Label) WHERE id(t) IN knn(…)`. HNSW: `CREATE INDEX name ON Label(prop) USING vector WITH (metric: 'cosine')`.
  - Release builds require env `GEODE_ALLOW_INSECURE_TLS=1` for `--insecure-tls-skip-verify` / `skip_verify`.
  - v0.8.5 `geode serve` has **no `--listen-grpc` flag** — QUIC only (works fine on x86). The compose files' `--listen-grpc` args must be removed/updated when running the server natively.
- DGX LLM endpoint: `http://curumo.shire-altair.ts.net:8000/v1`, model id `Qwen/Qwen3.8-27B`, max_model_len 262144, **Bearer auth required**. Key: obtain at runtime with `ssh curumo.shire-altair.ts.net "docker inspect vllm_node --format '{{range .Config.Env}}{{println .}}{{end}}' | grep '^VLLM_API_KEY=' | cut -d= -f2"`. Never write the key into any committed file — pass via env `VLLM_API_KEY`.
- Embeddings endpoint (after Task 7): `https://celebrimbor.shire-altair.ts.net:8001/v1/embeddings` (tailscale serve → localhost:8001), model `Qwen/Qwen3-Embedding-4B`, dimension 2560.
- Corpus: `haleth.shire-altair.ts.net:/data/kb/md` — 2,110 `.md` files, 5.3 GB. Stage locally before ingest.
- JIRA: project **ENG** on devnw.atlassian.net; custom field **GitLab Project = `customfield_10615`** (multi-select — fetch allowed options with `jira_get_field_options` before creating issues).
- No test suite exists in the playbooks repo. Every task must end with its stated verification commands passing (build + live smoke against a local `geode serve`).
- Local geode smoke-server recipe (reuse everywhere):
  ```bash
  GEODE_ADMIN_PASSWORD=geodesic geode serve --data-dir "$DATA" --listen 127.0.0.1:3141 &
  GEODE_ALLOW_INSECURE_TLS=1 geode query 'RETURN 1 AS ok' --server 127.0.0.1:3141 --insecure-tls-skip-verify -U geode -P geodesic
  ```

---

### Task 1: Sync repo with origin and create work branch

**Files:** none created; repo state only.

**Interfaces:**
- Produces: branch `feat/geode-085-upgrade` at `origin/main` (114 Renovate commits ahead of old HEAD `3fcb1e5`).

- [ ] **Step 1:** `cd /data/home/benji/src/devnw/codepros/geode/dgx-spark-playbooks && git fetch origin && git status --porcelain` — expect clean tree.
- [ ] **Step 2:** `git merge --ff-only origin/main` (verified HEAD is ancestor of origin/main).
- [ ] **Step 3:** `git checkout -b feat/geode-085-upgrade`.
- [ ] **Step 4:** Confirm Renovate majors landed: `grep -E '"next"|"typescript"|"tailwindcss"' nvidia/txt2kg/assets/frontend/package.json` — expect next 16.x, typescript ^7, tailwindcss ^4.
- [ ] **Step 5:** No commit needed (fast-forward only).

### Task 2: Vendor Node client v1.1.0 and make the txt2kg frontend build

**Files:**
- Create: `nvidia/txt2kg/assets/frontend/vendor/geodedb-client-1.1.0.tgz`
- Delete: `nvidia/txt2kg/assets/frontend/vendor/geodedb-client-1.0.0-alpha.20.tgz`
- Modify: `nvidia/txt2kg/assets/frontend/package.json` (dep path), `nvidia/txt2kg/assets/deploy/app/Dockerfile` (base image `node:18-slim` → `node:22-slim`; client engines require node ≥20)
- Modify (as needed): frontend sources to fix Renovate major-bump breakage (next 16 / ts 7 / tailwind 4 / langchain 1.x / zod 4)

**Interfaces:**
- Consumes: geode-client-nodejs tag v1.1.0 at `/data/home/benji/src/gl/gitlab.com/devnw/codepros/geode/geode-client-nodejs`.
- Produces: `pnpm build` succeeds; `@geodedb/client` v1.1.0 API available (`createClient`, `client.query(gql, { params })`, `queryAll`).

- [ ] **Step 1:** Build the tarball:
  ```bash
  cd /data/home/benji/src/gl/gitlab.com/devnw/codepros/geode/geode-client-nodejs
  git worktree add /tmp/claude-1000/**scratch**/gcn-v1.1.0 v1.1.0
  cd /tmp/claude-1000/**scratch**/gcn-v1.1.0 && npm ci && npm pack
  ```
  Copy the produced `geodedb-client-1.1.0.tgz` into `vendor/`; remove the alpha.20 tarball. Check whether the tarball bundles `geode.proto`; if not, refresh `vendor/geode-proto/geode.proto` from the v1.1.0 worktree and keep the Dockerfile COPY line.
- [ ] **Step 2:** Update `package.json`: `"@geodedb/client": "file:vendor/geodedb-client-1.1.0.tgz"`.
- [ ] **Step 3:** `pnpm install && pnpm build` in the frontend. Fix any breakage from the major bumps (mechanical migrations only; keep changes minimal). Iterate until build passes.
- [ ] **Step 4:** Update Dockerfile base images to `node:22-slim`.
- [ ] **Step 5:** Commit: `git commit -m "feat: upgrade @geodedb/client to v1.1.0, node 22, fix major-bump breakage"`.

### Task 3: Pin Python client v0.4.1 (chatbot backend + scripts)

**Files:**
- Modify: `nvidia/multi-agent-chatbot/assets/backend/pyproject.toml` (`@main` → `@v0.4.1`), regenerate `uv.lock`
- Modify: `nvidia/txt2kg/assets/scripts/requirements.txt` (`geode-client` → `geode-client @ git+https://gitlab.com/devnw/codepros/geode/geode-client-python.git@v0.4.1`)

**Interfaces:**
- Produces: `geode_client` v0.4.1 (`Client`, `TransportType`, `connection.query(gql, params)`, prepared statements, `batch_exec`).

- [ ] **Step 1:** Edit pyproject.toml dep to `geode-client @ git+https://gitlab.com/devnw/codepros/geode/geode-client-python.git@v0.4.1`.
- [ ] **Step 2:** `cd nvidia/multi-agent-chatbot/assets/backend && uv lock && uv sync` — expect geode-client 0.4.1 resolved.
- [ ] **Step 3:** `uv run python -c "from geode_client import Client, TransportType; print('ok')"`.
- [ ] **Step 4:** Update scripts/requirements.txt pin as above.
- [ ] **Step 5:** Commit: `git commit -m "feat: pin geode-client python to v0.4.1"`.

### Task 4: Modernize txt2kg `lib/geode.ts` for v0.8.5

**Files:**
- Modify: `nvidia/txt2kg/assets/frontend/lib/geode.ts` (all GQL construction)

**Interfaces:**
- Consumes: `@geodedb/client` v1.1.0 `query(gql, { params })`.
- Produces: unchanged public `GeodeService` method signatures (`addTriple`, `storeEntityEmbeddings`, `similaritySearch`, `storeDocumentChunk`, …) — callers in `lib/graph-db-service.ts` and API routes must not need changes.

Transformations (apply consistently to every query in the file):
1. Delete `vecLiteral()` and all `vectorf32(...)` string building. Pass embeddings as params: `SET e.embedding = $vec` / `{ params: { vec: embedding } }` (fixes #477/#478, removes injection surface).
2. Replace every string-interpolated value (names, ids, text) with `$param` bindings.
3. Relationship writes: keep `MERGE` for both nodes, then a **separate** `MATCH (s:Entity {name:$s}), (o:Entity {name:$o}) MERGE (s)-[:RELATIONSHIP {type:$p}]->(o)` statement (never chained MERGE — silent drop bug).
4. Similarity queries: prefer `WITH`-projection now (#476 fixed) and add a `knn`-based fast path:
   ```ts
   const rows = await client.queryAll(
     `MATCH (e:Entity) WHERE id(e) IN knn('embedding', $vec, $k, 'cosine')
      RETURN e.name AS name, vector_cosine(e.embedding, $vec) AS score ORDER BY score ASC`,
     { params: { vec, k } });
   ```
5. Add `ensureIndexes()` called once from the service init, creating HNSW indexes (swallow "already exists" errors):
   ```ts
   CREATE INDEX entity_emb_idx ON Entity(embedding) USING vector WITH (metric: 'cosine')
   CREATE INDEX chunk_emb_idx ON DocumentChunk(embedding) USING vector WITH (metric: 'cosine')
   ```
6. Remove stale comments referencing geode#476/#477/#478 except where the #478 param-binding note stays accurate.

- [ ] **Step 1:** Apply transformations; `pnpm build` passes.
- [ ] **Step 2:** Live smoke: start local geode (Global Constraints recipe), run a Node script (tsx) exercising `addTriple`, `storeEntityEmbeddings` with a **negative-first-element** embedding, `similaritySearch`, and re-run `addTriple` (idempotency: still exactly 1 relationship). All must pass.
- [ ] **Step 3:** Commit: `git commit -m "feat: geode v0.8.5 — param binding, knn/HNSW, drop fixed workarounds in txt2kg"`.

### Task 5: Modernize chatbot backend for v0.8.5

**Files:**
- Modify: `nvidia/multi-agent-chatbot/assets/backend/geode_storage.py`, `geode_vector_store.py`

**Interfaces:**
- Produces: unchanged public methods (`GeodeStorage.*`, `GeodeVectorStore.index_documents/get_documents`).

Same transformation set as Task 4 in Python: delete `_escape_gql_string`, use `await conn.query(gql, params)`, store embeddings as list params, `RagDocument` HNSW index `rag_emb_idx ON RagDocument(embedding) USING vector WITH (metric:'cosine')`, retrieval via `knn`, edge writes via separate MATCH+MERGE. Keep the `_val()` Value-unwrap helper (still needed).

- [ ] **Step 1:** Apply; `uv run python -m py_compile geode_storage.py geode_vector_store.py`.
- [ ] **Step 2:** Live smoke vs local geode: store 3 docs (one with negative-first embedding), `get_documents` returns ranked hits.
- [ ] **Step 3:** Commit: `git commit -m "feat: geode v0.8.5 param binding + knn in multi-agent-chatbot backend"`.

### Task 6: Remote-cluster config — vLLM API key support + DGX/celebrimbor overrides

**Files:**
- Modify: `nvidia/txt2kg/assets/frontend/lib/text-processor.ts`, `app/api/vllm/route.ts`, `app/api/extract-triples/route.ts`, `app/api/config/route.ts` (send `Authorization: Bearer ${process.env.VLLM_API_KEY}` when set)
- Modify: `nvidia/txt2kg/assets/frontend/lib/embeddings.ts` (honor `EMBEDDINGS_API_URL` pointing at an OpenAI-style `/v1/embeddings` endpoint + optional `EMBEDDINGS_MODEL_ID`; add OpenAI-format request branch if only the sentence-transformers custom format is implemented)
- Create: `nvidia/txt2kg/assets/deploy/compose/docker-compose.remote-cluster.yml`:
  ```yaml
  # Remote DGX cluster + celebrimbor encoder overlay.
  # Usage: docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.remote-cluster.yml up -d
  services:
    app:
      environment:
        - VLLM_BASE_URL=${VLLM_BASE_URL:-http://curumo.shire-altair.ts.net:8000/v1}
        - VLLM_MODEL=${VLLM_MODEL:-Qwen/Qwen3.8-27B}
        - VLLM_API_KEY=${VLLM_API_KEY}
        - OLLAMA_MODEL=disabled
        - EMBEDDINGS_API_URL=${EMBEDDINGS_API_URL:-https://celebrimbor.shire-altair.ts.net:8001}
        - EMBEDDINGS_MODEL_ID=${EMBEDDINGS_MODEL_ID:-Qwen/Qwen3-Embedding-4B}
        - GEODE_ALLOW_INSECURE_TLS=1
      extra_hosts: []
  ```
  plus disable the local `ollama`/`sentence-transformers` services in this overlay (`profiles: [disabled]` or scale 0 equivalent).
- Modify: compose geode service env to include `GEODE_ALLOW_INSECURE_TLS=1` where healthchecks use `--insecure-tls-skip-verify`.

- [ ] **Step 1:** Implement key plumbing + embeddings branch; `pnpm build`.
- [ ] **Step 2:** Verify with `VLLM_API_KEY=<runtime key> node` smoke: POST to DGX `/v1/chat/completions` through the same code path (tsx script) returns a completion.
- [ ] **Step 3:** Commit: `git commit -m "feat: remote DGX cluster + external encoder support (API key, OpenAI embeddings)"`.

### Task 7: Deploy encoder on celebrimbor (infra repo — do not commit)

**Files:**
- Modify: `/data/home/benji/src/devnw/infra/ansible-playbooks/inventory/devnw-static/host_vars/celebrimbor.yml` — replace the `vllm_embedding_enabled: false` block:
  ```yaml
  vllm_embedding_enabled: true
  vllm_embedding_model_repo: "Qwen/Qwen3-Embedding-4B"
  vllm_embedding_tensor_parallel: 1
  vllm_embedding_gpu_memory_utilization: 0.85
  vllm_embedding_max_model_len: 8192
  vllm_embedding_dimension: 2560
  vllm_embedding_require_hf_token: false
  vllm_embedding_extra_env:
    CUDA_VISIBLE_DEVICES: "0"
  ```

- [ ] **Step 1:** Fix NVML mismatch on celebrimbor (no GPU workloads currently running): `ssh celebrimbor 'sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia && nvidia-smi'`. If modules are held by the display manager: `sudo systemctl stop gdm` first, restart after. Only if reload is impossible, reboot (`sudo shutdown -r now`) and wait for the host + docker stack to return.
- [ ] **Step 2:** Edit host_vars as above.
- [ ] **Step 3:** `cd infra/ansible-playbooks && make deps` (if venv missing), then `make apply-check PLAYBOOK=playbooks/fleet/ml.yml INVENTORY=inventory/devnw-static/hosts.yml LIMIT=celebrimbor` (dry-run), then `make apply …` same args.
- [ ] **Step 4:** Verify: `curl -s https://celebrimbor.shire-altair.ts.net:8001/v1/models` lists the model; POST `/v1/embeddings` with `{"model":"Qwen/Qwen3-Embedding-4B","input":"hello"}` returns a 2560-dim vector.
- [ ] **Step 5:** Leave infra changes uncommitted; record diff in the final report.

### Task 8: Stage corpus + provision local geode

**Files:**
- Create: `~/datasets/kb-md/` (staged corpus)

- [ ] **Step 1:** `df -h /data/home` — need >12 GB free. `rsync -a --info=progress2 haleth.shire-altair.ts.net:/data/kb/md/ /data/home/benji/datasets/kb-md/` then `find … -name '*.md' | wc -l` == 2110.
- [ ] **Step 2:** Start persistent local geode: data dir `/data/home/benji/geode-kb-data`, `GEODE_ADMIN_PASSWORD=geodesic geode serve --data-dir /data/home/benji/geode-kb-data --listen 127.0.0.1:3141` under `nohup`/`setsid` with log file; verify `RETURN 1`.
- [ ] **Step 3:** Create HNSW indexes (entity_emb_idx, chunk_emb_idx) via `geode query`.

### Task 9: Headless ingestion driver + sample run (throughput gate)

**Files:**
- Create: `nvidia/txt2kg/assets/scripts/ingest_kb.py` — standalone asyncio driver using `geode_client` v0.4.1 directly (not the UI): for each `.md` file → chunk (recursive splitter, 2000 chars/200 overlap) → embed via celebrimbor (batched, 32 chunks/request) → store `DocumentChunk {docId, seq, text, embedding}` + `ProcessedDocument` marker (idempotent resume) → triple-extract via DGX Qwen (`/v1/chat/completions`, JSON triple prompt from `lib/text-processor.ts` semantics) → MERGE entities, MATCH+MERGE `RELATIONSHIP` edges. CLI: `--dir --server --limit N --concurrency --skip-triples --resume`. Env: `VLLM_API_KEY`, `EMBEDDINGS_API_URL`, `GEODE_ALLOW_INSECURE_TLS=1`. Log per-file status to `ingest_kb.log` + failures to `ingest_failures.jsonl`.

- [ ] **Step 1:** Write the driver; run on `--limit 10`.
- [ ] **Step 2:** Validate sample: chunk count >0, `MATCH (c:DocumentChunk) RETURN count(c)`, entities + relationships exist, `knn` query returns relevant chunks for a topical question.
- [ ] **Step 3:** Measure throughput; compute full-run ETA. Record in plan-execution notes. If triple extraction is the bottleneck (likely), full run proceeds with embeddings for **all** files and triples throttled at max sustainable concurrency — coverage reported honestly at the end.
- [ ] **Step 4:** Commit driver: `git commit -m "feat: headless kb ingestion driver for geode"`.

### Task 10: Bring up the full txt2kg system locally + full ingest

- [ ] **Step 1:** `make txt2kg`-equivalent with the remote overlay: `docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.remote-cluster.yml up -d` from `nvidia/txt2kg/assets/deploy/compose` with `VLLM_API_KEY` exported, geode service pointed at a bind-mount of `/data/home/benji/geode-kb-data` **or** app pointed at host geode via `GEODE_URL` (prefer host geode: set `network_mode` or `host.docker.internal` extra_host; document choice). Verify UI at `:3001` shows graph stats and can run a query against already-ingested sample data. If docker is unavailable locally, fall back to `pnpm build && pnpm start` for the frontend natively + host geode, and record that deviation.
- [ ] **Step 2:** Launch full ingest with `nohup`, resume-safe: all 2,110 files. Monitor via log; re-run failures once.
- [ ] **Step 3:** Final validation queries (counts of ProcessedDocument == 2110 minus recorded failures; DocumentChunk count; Entity/RELATIONSHIP counts; 3 topical `knn` searches return sensible text). Capture outputs for the report.

### Task 11: File geode bugs in JIRA ENG (GitLab Project field)

**Bugs to file (plus anything new found in Tasks 2–10):**
1. Chained `MERGE (a)…(b)… MERGE (a)-[:R]->(b)` silently creates no relationship (status 00000) — data-loss class. Repro from spike. GitLab project: geode.
2. #478 still open in v0.8.5: leading negative first element in list literal fails to parse (`42000 expected l_paren got minus`), including plain `RETURN [-0.1]`. GitLab project: geode.
3. `vectorf32($param)` rejected as "invalid builtin function call" while plain `$param` list works — inconsistent with literal usage. GitLab project: geode.
4. geodedb.com vector docs reference nonexistent functions (`vector_distance_cosine` et al.) which now hard-error `42883`, and HNSW `WITH` params `m`/`ef_construction`/`ef_search` are parsed-and-ignored. GitLab project: geodedb.com (docs) + geode (silent-ignore).
5. `geode serve` v0.8.5 lacks `--listen-grpc` though compose files and clients still reference gRPC DSNs (regression vs what playbooks used) — verify against geode repo before filing.

- [ ] **Step 1:** `jira_get_field_options` for `customfield_10615` in ENG; map options for `geode` and `geodedb.com` GitLab projects (create-fields check first via issue type Bug).
- [ ] **Step 2:** Search ENG for existing duplicates (`jira_search`: text match per bug) — update/comment instead of duplicating.
- [ ] **Step 3:** Create one Bug per item: repro steps, exact error output, version (`geode v0.8.5`), expected vs actual, workaround; set `customfield_10615` to the matching GitLab project option. Cross-reference the existing GitLab issue numbers (#424/#476/#477/#478) in descriptions.
- [ ] **Step 4:** Record created issue keys for the final report.

### Task 12: Finalize — push branch, write report

- [ ] **Step 1:** `git push -u origin feat/geode-085-upgrade` (playbooks repo only).
- [ ] **Step 2:** Final report: versions upgraded, workarounds removed/kept (and why), infra diff left uncommitted, ingest statistics, validation query outputs, JIRA keys filed, deviations.

## Self-Review Notes

- Spec coverage: update to latest geode+deps (T1–T3), leverage new features (T4–T6), run locally against DGX qwen (T6, T10), encoder on celebrimbor via ansible (T7), ingest corpus into KG with vectorization (T8–T10), JIRA bugs w/ GitLab Project field (T11). Covered.
- Corpus path note: user said `/data/datasets/docs/md`; that path exists nowhere on the fleet — the real corpus is `haleth:/data/kb/md` (2,110 md files). Flag in final report.
- Type consistency: `GeodeService` and `GeodeVectorStore` public signatures deliberately frozen (T4/T5) so routes need no changes.
