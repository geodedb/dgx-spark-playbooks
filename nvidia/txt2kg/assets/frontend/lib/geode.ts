//
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 CodePros. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

/**
 * Unified Geode service replacing ArangoDB, Neo4j, Qdrant, and Pinecone.
 * Uses @geodedb/client for graph + vector operations in a single database.
 *
 * All values (names, ids, text, embeddings) are passed as query parameters
 * ($param bindings) — never string-interpolated into GQL. Embeddings in
 * particular must be bound as params: literal lists with a leading negative
 * element still fail to parse (geode#478), and param binding sidesteps the
 * issue entirely while removing the injection surface.
 *
 * Embedding storage uses `vectorf32(toString($vec))` with $vec bound to the
 * JSON string of the number[] (see vecParam). This is deliberate: a plain
 * list param stores a LIST-typed property that vector_cosine() accepts but
 * knn() and the HNSW vector indexes ignore entirely (verified against the
 * geode v0.8.5 source: both the index and scan knn paths only consider
 * VectorF32/VectorI32-typed properties). `vectorf32($vec)` is rejected by
 * the parser ("invalid builtin function call" — params are not literal
 * arguments), but a nested call is allowed, so `vectorf32(toString($vec))`
 * parses and evaluates the param server-side into a real VECTORF32 value.
 * The data path stays fully parameterized: no injection, no #477/#478
 * literal-parsing issues.
 *
 * QUERY vectors (the argument to knn()/vector_cosine()) are bound as plain
 * number[] list params — those are accepted directly.
 */
import { createClient, GeodeClient, isDriverError } from '@geodedb/client';

export interface DocumentSearchResult {
  id: string;
  score: number;
  metadata?: Record<string, any>;
}

/** Coerce a caller-supplied limit into a safe positive integer for LIMIT clauses. */
function safeLimit(n: number, fallback: number): number {
  const v = Math.floor(Number(n));
  return Number.isFinite(v) && v > 0 ? v : fallback;
}

/**
 * Serialize an embedding for storage via `vectorf32(toString($vec))`.
 * JSON list-of-numbers form is exactly what vectorf32's string parser
 * accepts (scientific notation included).
 */
function vecParam(v: number[]): string {
  return JSON.stringify(v);
}

/**
 * knn() candidate over-fetch settings — see the `knnRows` doc comment for
 * why the boost exists. A multiplier this small is cheap even for large
 * graphs; the cap keeps a huge `k` from forcing an unbounded index scan.
 */
const KNN_BOOST_FACTOR = 8;
const KNN_BOOST_CAP = 512;

/** Requested candidate count for knn(), boosted to survive cross-label crowding (see knnRows). */
function knnBoost(k: number): number {
  return Math.min(k * KNN_BOOST_FACTOR, KNN_BOOST_CAP);
}

export class GeodeService {
  private client: GeodeClient | null = null;
  private static instance: GeodeService;
  private initialized: boolean = false;
  private isInitializing = false;
  private indexesEnsured = false;

  private constructor() {}

  public static getInstance(): GeodeService {
    if (!GeodeService.instance) {
      GeodeService.instance = new GeodeService();
    }
    return GeodeService.instance;
  }

  /**
   * Initialize the Geode connection
   */
  public async initialize(url?: string): Promise<void> {
    if (this.initialized || this.isInitializing) {
      return;
    }

    this.isInitializing = true;

    try {
      const connectionUrl = url || process.env.GEODE_URL || 'quic://localhost:3141?insecure_tls_skip_verify=true';
      console.log(`Connecting to Geode at ${connectionUrl}`);

      this.client = await createClient(connectionUrl);
      await this.client.ping();

      this.initialized = true;
      this.isInitializing = false;
      console.log('Geode initialized successfully');

      await this.ensureIndexes();
    } catch (error) {
      this.isInitializing = false;
      console.error('Failed to initialize Geode:', error);
      throw error;
    }
  }

  public isInitialized(): boolean {
    return this.initialized;
  }

  private async ensureInitialized(): Promise<void> {
    if (!this.initialized) {
      await this.initialize();
    }
    if (!this.client) {
      throw new Error('Geode connection not initialized. Call initialize() first.');
    }
  }

  /**
   * Create the HNSW vector indexes used by knn() retrieval. Idempotent:
   * "index already exists" errors are recognized and ignored; any other
   * error is surfaced as a warning (retrieval still works via full scans).
   */
  private async ensureIndexes(): Promise<void> {
    if (this.indexesEnsured || !this.client) {
      return;
    }

    const ddl = [
      `CREATE INDEX entity_emb_idx ON Entity(embedding) USING vector WITH (metric: 'cosine')`,
      `CREATE INDEX chunk_emb_idx ON DocumentChunk(embedding) USING vector WITH (metric: 'cosine')`,
    ];

    for (const stmt of ddl) {
      try {
        await this.client.exec(stmt);
      } catch (error) {
        if (this.isIndexExistsError(error)) {
          continue; // index already present — expected on restart
        }
        console.warn('[Geode] Failed to create vector index (continuing without):', error);
      }
    }

    this.indexesEnsured = true;
  }

  /** True when the error indicates the index already exists. */
  private isIndexExistsError(error: unknown): boolean {
    if (isDriverError(error) && error.statusClass === '42P07') {
      return true;
    }
    const msg = error instanceof Error ? error.message : String(error);
    return /already exists|duplicate/i.test(msg);
  }

  /**
   * Close the Geode connection
   */
  public async close(): Promise<void> {
    if (this.client) {
      await this.client.close();
      this.client = null;
      this.initialized = false;
      this.indexesEnsured = false;
      console.log('Geode connection closed');
    }
  }

  // ─── Graph Operations (replaces ArangoDB / Neo4j) ────────────────────

  /**
   * Import subject-predicate-object triples into the graph
   */
  public async importTriples(
    triples: { subject: string; predicate: string; object: string }[]
  ): Promise<void> {
    await this.ensureInitialized();

    const batchSize = 500;
    let imported = 0;

    for (let i = 0; i < triples.length; i += batchSize) {
      const batch = triples.slice(i, i + batchSize);

      for (const t of batch) {
        const subject = t.subject.trim();
        const predicate = t.predicate.trim();
        const object = t.object.trim();
        if (!subject || !predicate || !object) continue;

        try {
          // MERGE both nodes first, then MERGE the relationship in a
          // SEPARATE statement. Chaining node MERGEs with a relationship
          // MERGE in one statement silently creates no relationship in
          // Geode — never combine them.
          await this.client!.exec(
            `MERGE (s:Entity {name: $s})`,
            { params: { s: subject } }
          );
          await this.client!.exec(
            `MERGE (o:Entity {name: $o})`,
            { params: { o: object } }
          );
          await this.client!.exec(
            `MATCH (s:Entity {name: $s}), (o:Entity {name: $o})
             MERGE (s)-[:RELATIONSHIP {type: $p}]->(o)`,
            { params: { s: subject, o: object, p: predicate } }
          );
        } catch (err) {
          console.warn('Error importing triple, skipping:', err);
        }
      }

      imported += batch.length;
      console.log(`[Geode] Imported ${imported}/${triples.length} triples`);
    }

    console.log(`Successfully imported ${triples.length} triples into Geode`);
  }

  /**
   * Get graph data (nodes + relationships) for visualization
   */
  public async getGraphData(): Promise<{
    nodes: Array<{ id: string; labels: string[]; name: string; [key: string]: any }>;
    relationships: Array<{ id: string; source: string; target: string; type: string; [key: string]: any }>;
  }> {
    await this.ensureInitialized();

    try {
      // toString(id(..)): geode node ids exceed Number.MAX_SAFE_INTEGER, and
      // raw id() projections fail to decode client-side — project as strings.
      const entityRows = await this.client!.queryAll(
        `MATCH (e:Entity) RETURN e.name AS name, toString(id(e)) AS nodeId`
      );

      const nodes = entityRows.map((row: any) => ({
        id: String(row.nodeId),
        labels: ['Entity'],
        name: row.name,
      }));

      const relRows = await this.client!.queryAll(
        `MATCH (s:Entity)-[r:RELATIONSHIP]->(o:Entity)
         RETURN toString(id(s)) AS sourceId, toString(id(o)) AS targetId, r.type AS type, toString(id(r)) AS relId`
      );

      const relationships = relRows.map((row: any) => ({
        id: String(row.relId),
        source: String(row.sourceId),
        target: String(row.targetId),
        type: row.type,
      }));

      return { nodes, relationships };
    } catch (error) {
      console.error('Error getting graph data from Geode:', error);
      throw error;
    }
  }

  /**
   * Graph traversal with keyword matching
   */
  public async graphTraversal(
    keywords: string[],
    maxDepth: number = 2,
    maxResults: number = 100
  ): Promise<Array<{
    subject: string;
    predicate: string;
    object: string;
    confidence: number;
    depth?: number;
  }>> {
    await this.ensureInitialized();

    const filteredKeywords = keywords.filter(kw => kw.length > 2).map(kw => kw.toLowerCase());
    if (filteredKeywords.length === 0) return [];

    console.log(`[Geode] graphTraversal with keywords: ${filteredKeywords.join(', ')}`);

    try {
      const allResults: Array<{ subject: string; predicate: string; object: string; confidence: number; depth: number }> = [];
      const seen = new Set<string>();

      for (const keyword of filteredKeywords) {
        const rows = await this.client!.queryAll(
          `MATCH (s:Entity)-[r:RELATIONSHIP]->(o:Entity)
           WHERE s.name CONTAINS $kw OR o.name CONTAINS $kw OR r.type CONTAINS $kw
           RETURN DISTINCT s.name AS subject, r.type AS predicate, o.name AS object`,
          { params: { kw: keyword } }
        );

        for (const row of rows) {
          const key = `${row.subject}|${row.predicate}|${row.object}`;
          if (!seen.has(key)) {
            seen.add(key);
            allResults.push({
              subject: row.subject as string,
              predicate: row.predicate as string,
              object: row.object as string,
              confidence: 1.0 / (1.0 + allResults.length * 0.01),
              depth: 0,
            });
          }
        }
      }

      const sorted = allResults
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, maxResults);

      console.log(`[Geode] Found ${sorted.length} triples for keywords: ${keywords.join(', ')}`);
      return sorted;
    } catch (error) {
      console.error('Error performing graph traversal in Geode:', error);
      throw error;
    }
  }

  /**
   * Clear all data from the database
   */
  public async clearDatabase(): Promise<void> {
    await this.ensureInitialized();

    try {
      await this.client!.exec('MATCH (n) DETACH DELETE n');
      console.log('Geode database cleared successfully');
    } catch (error) {
      console.error('Error clearing Geode database:', error);
      throw error;
    }
  }

  // ─── Vector Operations (replaces Qdrant / Pinecone) ──────────────────

  /**
   * Store entity embeddings as VECTORF32 properties (param-bound; see the
   * module header for why storage goes through vectorf32(toString($vec))).
   */
  public async storeEmbeddings(
    entityEmbeddings: Map<string, number[]>,
    textContentMap?: Map<string, string>
  ): Promise<void> {
    await this.ensureInitialized();

    try {
      let stored = 0;
      for (const [entityName, embedding] of entityEmbeddings.entries()) {
        await this.client!.exec(
          `MERGE (e:Entity {name: $name})
           SET e.embedding = vectorf32(toString($vec))`,
          { params: { name: entityName, vec: vecParam(embedding) } }
        );
        stored++;
      }
      console.log(`Stored ${stored} entity embeddings in Geode`);
    } catch (error) {
      console.log('Error storing embeddings in Geode:', error);
    }
  }

  /**
   * Store embeddings with metadata
   */
  public async storeEmbeddingsWithMetadata(
    embeddings: Map<string, number[]>,
    textContent: Map<string, string>,
    metadata: Map<string, any>
  ): Promise<void> {
    await this.ensureInitialized();

    try {
      let stored = 0;
      for (const [key, embedding] of embeddings.entries()) {
        await this.client!.exec(
          `MERGE (e:Entity {name: $name})
           SET e.embedding = vectorf32(toString($vec)), e.text = $text`,
          { params: { name: key, vec: vecParam(embedding), text: textContent.get(key) || '' } }
        );
        stored++;
      }
      console.log(`Stored ${stored} embeddings with metadata in Geode`);
    } catch (error) {
      console.log('Error storing embeddings with metadata in Geode:', error);
    }
  }

  /**
   * Shared knn()-with-fallback query executor for both Entity and
   * DocumentChunk vector lookups.
   *
   * IMPORTANT — cross-label crowding: `entity_emb_idx` (on Entity.embedding)
   * and `chunk_emb_idx` (on DocumentChunk.embedding) both index the property
   * NAME `embedding`, and `knn('embedding', $vec, $k, 'cosine')` selects its
   * candidate set by property name across ALL labels *before* the label
   * filter in MATCH is applied. With only `k` candidates requested, Entity
   * and DocumentChunk nodes crowd each other's candidate list, so a
   * label-filtered query can come back with fewer than k rows even when k
   * matching nodes exist. Renaming the property per label was considered
   * and rejected — it would break existing data and the ingestion schema.
   * Instead every knn() call site over-fetches: it asks knn() for
   * `kBoost = k * KNN_BOOST_FACTOR` (capped at KNN_BOOST_CAP) candidates,
   * then trims back to `k` with LIMIT after the label filter has narrowed
   * the candidate set down to matching nodes.
   *
   * `label` is the node label to match, `varName` the MATCH variable, and
   * `returnProjection` the RETURN clause — it must reference a `score`
   * column bound via a preceding `WITH ${varName}, vector_cosine(...) AS
   * score` (both the fast and fallback paths bind it that way so one
   * projection string works for both, preserving identical semantics).
   * Uses the HNSW index when one covers the property, otherwise the server
   * falls back to a full scan. vector_cosine returns cosine SIMILARITY
   * (1 = identical, 0 = orthogonal, -1 = opposite; verified on v0.8.5), so
   * rows are ordered descending. Scores can arrive as decimal strings —
   * callers must Number() them.
   */
  private async knnRows(
    label: 'Entity' | 'DocumentChunk',
    varName: string,
    returnProjection: string,
    queryEmbedding: number[],
    k: number
  ): Promise<Array<Record<string, unknown>>> {
    const limit = safeLimit(k, 10);
    const kBoost = knnBoost(limit);
    try {
      return await this.client!.queryAll(
        `MATCH (${varName}:${label})
         WHERE id(${varName}) IN knn('embedding', $vec, $kBoost, 'cosine')
         WITH ${varName}, vector_cosine(${varName}.embedding, $vec) AS score
         RETURN ${returnProjection}
         ORDER BY score DESC
         LIMIT ${limit}`,
        { params: { vec: queryEmbedding, kBoost } }
      );
    } catch (error) {
      // Fallback: full-scan WITH projection.
      console.warn(`[Geode] knn() fast path unavailable for ${label}, falling back to full scan:`, error);
      return await this.client!.queryAll(
        `MATCH (${varName}:${label})
         WHERE ${varName}.embedding IS NOT NULL
         WITH ${varName}, vector_cosine(${varName}.embedding, $vec) AS score
         RETURN ${returnProjection}
         ORDER BY score DESC
         LIMIT ${limit}`,
        { params: { vec: queryEmbedding } }
      );
    }
  }

  /** knn()-backed nearest-entity lookup; see knnRows for the over-fetch rationale. */
  private async knnEntityRows(
    queryEmbedding: number[],
    k: number
  ): Promise<Array<Record<string, unknown>>> {
    return this.knnRows(
      'Entity',
      'e',
      'e.name AS name, score, e.text AS text',
      queryEmbedding,
      k
    );
  }

  /**
   * Find entities similar to a query embedding using vector cosine similarity
   */
  public async findSimilarEntities(queryEmbedding: number[], topK: number = 10): Promise<string[]> {
    await this.ensureInitialized();

    try {
      const rows = await this.knnEntityRows(queryEmbedding, safeLimit(topK, 10));
      return rows
        .map((r: any) => ({ name: r.name as string, score: Number(r.score) }))
        .filter(r => r.score > 0.1)
        .map(r => r.name);
    } catch (error) {
      console.log('Error finding similar entities:', error);
      return [];
    }
  }

  /**
   * Find similar entities with metadata
   */
  public async findSimilarEntitiesWithMetadata(
    embedding: number[],
    limit: number = 10
  ): Promise<{ entities: string[]; metadata: Map<string, any> }> {
    await this.ensureInitialized();

    try {
      const rows = await this.knnEntityRows(embedding, safeLimit(limit, 10));

      const scored = rows
        .map((r: any) => ({ name: r.name as string, score: Number(r.score), text: r.text }))
        .filter(r => r.score > 0.1);

      const entities = scored.map(r => r.name);
      const metadataMap = new Map<string, any>();
      for (const r of scored) {
        metadataMap.set(r.name, {
          score: r.score,
          text: r.text,
          originalId: r.name,
        });
      }
      return { entities, metadata: metadataMap };
    } catch (error) {
      console.log('Error finding similar entities with metadata:', error);
      return { entities: [], metadata: new Map() };
    }
  }

  /**
   * Store document chunks for Pure RAG
   */
  public async storeDocumentChunks(
    documents: string[],
    embeddings: number[][],
    metadata?: Record<string, any>[]
  ): Promise<void> {
    await this.ensureInitialized();

    try {
      console.log(`Storing ${documents.length} document chunks in Geode`);

      for (let i = 0; i < documents.length; i++) {
        await this.client!.exec(
          `CREATE (d:DocumentChunk {
            text: $text,
            embedding: vectorf32(toString($vec)),
            source: $source,
            createdAt: $createdAt
          })`,
          {
            params: {
              text: documents[i],
              vec: vecParam(embeddings[i]),
              source: metadata?.[i]?.source || 'unknown',
              createdAt: new Date().toISOString(),
            },
          }
        );
      }

      console.log(`Stored ${documents.length} document chunks in Geode`);
    } catch (error) {
      console.log('Error storing document chunks:', error);
    }
  }

  /**
   * Find similar documents using vector cosine similarity
   */
  public async findSimilarDocuments(
    queryEmbedding: number[],
    topK: number = 10
  ): Promise<DocumentSearchResult[]> {
    await this.ensureInitialized();

    const k = safeLimit(topK, 10);
    try {
      const rows = await this.knnRows(
        'DocumentChunk',
        'd',
        'd.text AS text, d.source AS source, score, toString(id(d)) AS docId',
        queryEmbedding,
        k
      );

      return rows
        .map((r: any) => ({ row: r, score: Number(r.score) }))
        .filter(({ score }) => score > 0.1)
        .map(({ row, score }) => ({
          id: String(row.docId),
          score,
          metadata: {
            text: row.text,
            source: row.source,
            originalId: String(row.docId),
          },
        }));
    } catch (error) {
      console.log('Error finding similar documents:', error);
      return [];
    }
  }

  /**
   * Delete all entities from the vector store (clear embeddings)
   */
  public async deleteAllEntities(): Promise<boolean> {
    await this.ensureInitialized();

    try {
      await this.client!.exec(
        `MATCH (e:Entity) WHERE e.embedding IS NOT NULL REMOVE e.embedding`
      );
      await this.client!.exec(
        `MATCH (d:DocumentChunk) DETACH DELETE d`
      );
      console.log('Deleted all vector data from Geode');
      return true;
    } catch (error) {
      console.log('Error deleting vector data:', error);
      return false;
    }
  }

  // ─── Logging & Tracking ──────────────────────────────────────────────

  /**
   * Log query information and metrics
   */
  public async logQuery(
    query: string,
    queryMode: 'traditional' | 'vector-search' | 'pure-rag',
    metrics: {
      executionTimeMs: number;
      relevanceScore?: number;
      precision?: number;
      recall?: number;
      resultCount: number;
    }
  ): Promise<void> {
    await this.ensureInitialized();

    try {
      await this.client!.exec(
        `CREATE (l:QueryLog {
          query: $query,
          queryMode: $queryMode,
          executionTimeMs: $executionTimeMs,
          relevanceScore: $relevanceScore,
          precision: $precision,
          recall: $recall,
          resultCount: $resultCount,
          timestamp: $timestamp
        })`,
        {
          params: {
            query,
            queryMode,
            executionTimeMs: metrics.executionTimeMs,
            relevanceScore: metrics.relevanceScore ?? 0,
            precision: metrics.precision ?? 0,
            recall: metrics.recall ?? 0,
            resultCount: metrics.resultCount,
            timestamp: new Date().toISOString(),
          },
        }
      );
    } catch (error) {
      console.error('Error logging query to Geode:', error);
    }
  }

  /**
   * Get query logs
   */
  public async getQueryLogs(limit: number = 100): Promise<any[]> {
    await this.ensureInitialized();

    try {
      const rows = await this.client!.queryAll(
        `MATCH (l:QueryLog)
         RETURN l.query AS query, l.queryMode AS queryMode,
                l.executionTimeMs AS executionTimeMs,
                l.relevanceScore AS relevanceScore,
                l.precision AS precision, l.recall AS recall,
                l.resultCount AS resultCount, l.timestamp AS timestamp
         ORDER BY l.timestamp DESC
         LIMIT ${safeLimit(limit, 100)}`
      );
      return rows;
    } catch (error) {
      console.error('Error getting query logs from Geode:', error);
      return [];
    }
  }

  // ─── Document Tracking ───────────────────────────────────────────────

  /**
   * Check if a document has been processed
   */
  public async isDocumentProcessed(documentName: string): Promise<boolean> {
    await this.ensureInitialized();

    try {
      const rows = await this.client!.queryAll(
        `MATCH (d:ProcessedDocument {documentName: $name})
         RETURN d.documentName AS name LIMIT 1`,
        { params: { name: documentName.trim() } }
      );
      return rows.length > 0;
    } catch (error) {
      return false;
    }
  }

  /**
   * Mark a document as processed
   */
  public async markDocumentAsProcessed(documentName: string, tripleCount: number): Promise<void> {
    await this.ensureInitialized();

    try {
      await this.client!.exec(
        `MERGE (d:ProcessedDocument {documentName: $name})
         SET d.tripleCount = $tripleCount, d.processedAt = $processedAt`,
        {
          params: {
            name: documentName.trim(),
            tripleCount,
            processedAt: new Date().toISOString(),
          },
        }
      );
      console.log(`Marked document "${documentName}" as processed with ${tripleCount} triples`);
    } catch (error) {
      console.error('Error marking document as processed:', error);
    }
  }

  /**
   * Get all processed documents
   */
  public async getProcessedDocuments(): Promise<string[]> {
    await this.ensureInitialized();

    try {
      const rows = await this.client!.queryAll(
        `MATCH (d:ProcessedDocument) RETURN d.documentName AS name`
      );
      return rows.map((r: any) => r.name as string);
    } catch (error) {
      return [];
    }
  }

  // ─── Stats ───────────────────────────────────────────────────────────

  /**
   * Get database statistics
   */
  public async getStats(): Promise<any> {
    await this.ensureInitialized();

    try {
      const entityCount = await this.client!.queryFirst(
        `MATCH (e:Entity) RETURN count(e) AS count`
      );
      const embeddingCount = await this.client!.queryFirst(
        `MATCH (e:Entity) WHERE e.embedding IS NOT NULL RETURN count(e) AS count`
      );
      const docChunkCount = await this.client!.queryFirst(
        `MATCH (d:DocumentChunk) RETURN count(d) AS count`
      );
      const relCount = await this.client!.queryFirst(
        `MATCH ()-[r:RELATIONSHIP]->() RETURN count(r) AS count`
      );

      return {
        totalVectorCount: Number((embeddingCount as any)?.count || 0) + Number((docChunkCount as any)?.count || 0),
        entityCount: Number((entityCount as any)?.count || 0),
        embeddingCount: Number((embeddingCount as any)?.count || 0),
        documentChunkCount: Number((docChunkCount as any)?.count || 0),
        relationshipCount: Number((relCount as any)?.count || 0),
        source: 'geode',
        httpHealthy: true,
      };
    } catch (error) {
      return {
        totalVectorCount: 0,
        source: 'error',
        httpHealthy: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  /**
   * Get driver info
   */
  public getDriverInfo(): Record<string, any> {
    if (!this.client) {
      return { status: 'not connected' };
    }
    return {
      status: 'connected',
      type: 'geode',
    };
  }

  // ─── Collection Management (Qdrant compat) ──────────────────────────

  /**
   * Create collection — no-op for Geode
   */
  public async createCollection(): Promise<void> {
    await this.ensureInitialized();
    console.log('Geode: create collection is a no-op — graph nodes are created dynamically');
  }

  /**
   * Upsert vectors (Qdrant compat)
   */
  public async upsertVectors(points: Array<{ id: string; vector: number[]; payload?: Record<string, any> }>): Promise<boolean> {
    await this.ensureInitialized();

    try {
      for (const point of points) {
        const text = point.payload?.text || '';
        const originalId = point.payload?.originalId || point.id;

        if (point.payload?.type === 'document') {
          await this.client!.exec(
            `CREATE (d:DocumentChunk {
              text: $text,
              embedding: vectorf32(toString($vec)),
              originalId: $originalId,
              createdAt: $createdAt
            })`,
            {
              params: {
                text,
                vec: vecParam(point.vector),
                originalId,
                createdAt: new Date().toISOString(),
              },
            }
          );
        } else {
          await this.client!.exec(
            `MERGE (e:Entity {name: $name})
             SET e.embedding = vectorf32(toString($vec)), e.text = $text`,
            { params: { name: originalId, vec: vecParam(point.vector), text } }
          );
        }
      }
      return true;
    } catch (error) {
      console.log('Error upserting vectors:', error);
      return false;
    }
  }

  /**
   * Get all entities
   */
  public async getAllEntities(limit: number = 1000): Promise<string[]> {
    await this.ensureInitialized();

    try {
      const rows = await this.client!.queryAll(
        `MATCH (e:Entity) RETURN e.name AS name LIMIT ${safeLimit(limit, 1000)}`
      );
      return rows.map((r: any) => r.name as string);
    } catch (error) {
      return [];
    }
  }

  /**
   * Delete specific entities
   */
  public async deleteEntities(entityNames: string[]): Promise<boolean> {
    await this.ensureInitialized();

    try {
      for (const name of entityNames) {
        await this.client!.exec(
          `MATCH (e:Entity {name: $name}) DETACH DELETE e`,
          { params: { name } }
        );
      }
      return true;
    } catch (error) {
      console.log('Error deleting entities:', error);
      return false;
    }
  }
}
