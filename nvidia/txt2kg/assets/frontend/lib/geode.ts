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
 */
import { createClient, GeodeClient } from '@geodedb/client';

export interface DocumentSearchResult {
  id: string;
  score: number;
  metadata?: Record<string, any>;
}

export class GeodeService {
  private client: GeodeClient | null = null;
  private static instance: GeodeService;
  private initialized: boolean = false;
  private isInitializing = false;

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
   * Close the Geode connection
   */
  public async close(): Promise<void> {
    if (this.client) {
      await this.client.close();
      this.client = null;
      this.initialized = false;
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

        const escaped = (s: string) => s.replace(/'/g, "\\'");
        try {
          // MERGE nodes first, then MATCH+CREATE the relationship
          // (Geode does not support MERGE for relationship creation)
          await this.client!.exec(
            `MERGE (s:Entity {name: '${escaped(subject)}'})`
          );
          await this.client!.exec(
            `MERGE (o:Entity {name: '${escaped(object)}'})`
          );
          await this.client!.exec(
            `MATCH (s:Entity {name: '${escaped(subject)}'}), (o:Entity {name: '${escaped(object)}'})
             CREATE (s)-[:RELATIONSHIP {type: '${escaped(predicate)}'}]->(o)`
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
      const entityRows = await this.client!.queryAll(
        `MATCH (e:Entity) RETURN e.name AS name, id(e) AS nodeId`
      );

      const nodes = entityRows.map((row: any) => ({
        id: String(row.nodeId),
        labels: ['Entity'],
        name: row.name,
      }));

      const relRows = await this.client!.queryAll(
        `MATCH (s:Entity)-[r:RELATIONSHIP]->(o:Entity)
         RETURN id(s) AS sourceId, id(o) AS targetId, r.type AS type, id(r) AS relId`
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
        const escaped = keyword.replace(/'/g, "\\'");
        const rows = await this.client!.queryAll(
          `MATCH (s:Entity)-[r:RELATIONSHIP]->(o:Entity)
           WHERE s.name CONTAINS '${escaped}' OR o.name CONTAINS '${escaped}' OR r.type CONTAINS '${escaped}'
           RETURN DISTINCT s.name AS subject, r.type AS predicate, o.name AS object`
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
   * Store entity embeddings as vectorf32 properties
   */
  public async storeEmbeddings(
    entityEmbeddings: Map<string, number[]>,
    textContentMap?: Map<string, string>
  ): Promise<void> {
    await this.ensureInitialized();

    try {
      let stored = 0;
      for (const [entityName, embedding] of entityEmbeddings.entries()) {
        const embStr = `[${embedding.join(',')}]`;
        const escaped = entityName.replace(/'/g, "\\'");
        await this.client!.exec(
          `MERGE (e:Entity {name: '${escaped}'})
           ON CREATE SET e.embedding = vectorf32(${embStr})
           ON MATCH SET e.embedding = vectorf32(${embStr})`
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
        const embStr = `[${embedding.join(',')}]`;
        const text = (textContent.get(key) || '').replace(/'/g, "\\'");
        const escaped = key.replace(/'/g, "\\'");
        await this.client!.exec(
          `MERGE (e:Entity {name: '${escaped}'})
           ON CREATE SET e.embedding = vectorf32(${embStr}), e.text = '${text}'
           ON MATCH SET e.embedding = vectorf32(${embStr}), e.text = '${text}'`
        );
        stored++;
      }
      console.log(`Stored ${stored} embeddings with metadata in Geode`);
    } catch (error) {
      console.log('Error storing embeddings with metadata in Geode:', error);
    }
  }

  /**
   * Find entities similar to a query embedding using vector cosine similarity
   */
  public async findSimilarEntities(queryEmbedding: number[], topK: number = 10): Promise<string[]> {
    await this.ensureInitialized();

    try {
      const embStr = `[${queryEmbedding.join(',')}]`;
      const rows = await this.client!.queryAll(
        `MATCH (e:Entity)
         WHERE e.embedding IS NOT NULL
         WITH e, vector_cosine(e.embedding, vectorf32(${embStr})) AS score
         WHERE score > 0.1
         RETURN e.name AS name
         ORDER BY score DESC
         LIMIT ${topK}`
      );

      return rows.map((r: any) => r.name as string);
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
      const embStr = `[${embedding.join(',')}]`;
      const rows = await this.client!.queryAll(
        `MATCH (e:Entity)
         WHERE e.embedding IS NOT NULL
         WITH e, vector_cosine(e.embedding, vectorf32(${embStr})) AS score
         WHERE score > 0.1
         RETURN e.name AS name, score, e.text AS text
         ORDER BY score DESC
         LIMIT ${limit}`
      );

      const entities = rows.map((r: any) => r.name as string);
      const metadataMap = new Map<string, any>();
      for (const r of rows) {
        metadataMap.set(r.name as string, {
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
        const text = documents[i].replace(/\\/g, '\\\\').replace(/'/g, "\\'");
        const embStr = `[${embeddings[i].join(',')}]`;
        const source = (metadata?.[i]?.source || 'unknown').replace(/'/g, "\\'");

        await this.client!.exec(
          `CREATE (d:DocumentChunk {
            text: '${text}',
            embedding: vectorf32(${embStr}),
            source: '${source}',
            createdAt: '${new Date().toISOString()}'
          })`
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

    try {
      const embStr = `[${queryEmbedding.join(',')}]`;
      const rows = await this.client!.queryAll(
        `MATCH (d:DocumentChunk)
         WHERE d.embedding IS NOT NULL
         WITH d, vector_cosine(d.embedding, vectorf32(${embStr})) AS score
         WHERE score > 0.1
         RETURN d.text AS text, d.source AS source, score, id(d) AS docId
         ORDER BY score DESC
         LIMIT ${topK}`
      );

      return rows.map((r: any) => ({
        id: String(r.docId),
        score: r.score as number,
        metadata: {
          text: r.text,
          source: r.source,
          originalId: String(r.docId),
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
      const escaped = query.replace(/'/g, "\\'");
      await this.client!.exec(
        `CREATE (l:QueryLog {
          query: '${escaped}',
          queryMode: '${queryMode}',
          executionTimeMs: ${metrics.executionTimeMs},
          relevanceScore: ${metrics.relevanceScore ?? 0},
          precision: ${metrics.precision ?? 0},
          recall: ${metrics.recall ?? 0},
          resultCount: ${metrics.resultCount},
          timestamp: '${new Date().toISOString()}'
        })`
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
         LIMIT ${limit}`
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
      const escaped = documentName.trim().replace(/'/g, "\\'");
      const rows = await this.client!.queryAll(
        `MATCH (d:ProcessedDocument {documentName: '${escaped}'})
         RETURN d.documentName AS name LIMIT 1`
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
      const escaped = documentName.trim().replace(/'/g, "\\'");
      const now = new Date().toISOString();
      await this.client!.exec(
        `MERGE (d:ProcessedDocument {documentName: '${escaped}'})
         ON CREATE SET d.tripleCount = ${tripleCount}, d.processedAt = '${now}'
         ON MATCH SET d.tripleCount = ${tripleCount}, d.processedAt = '${now}'`
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
        const text = (point.payload?.text || '').replace(/'/g, "\\'");
        const originalId = (point.payload?.originalId || point.id).replace(/'/g, "\\'");
        const embStr = `[${point.vector.join(',')}]`;

        if (point.payload?.type === 'document') {
          await this.client!.exec(
            `CREATE (d:DocumentChunk {
              text: '${text}',
              embedding: vectorf32(${embStr}),
              originalId: '${originalId}',
              createdAt: '${new Date().toISOString()}'
            })`
          );
        } else {
          await this.client!.exec(
            `MERGE (e:Entity {name: '${originalId}'})
             ON CREATE SET e.embedding = vectorf32(${embStr}), e.text = '${text}'
             ON MATCH SET e.embedding = vectorf32(${embStr}), e.text = '${text}'`
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
        `MATCH (e:Entity) RETURN e.name AS name LIMIT ${limit}`
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
        const escaped = name.replace(/'/g, "\\'");
        await this.client!.exec(
          `MATCH (e:Entity {name: '${escaped}'}) DETACH DELETE e`
        );
      }
      return true;
    } catch (error) {
      console.log('Error deleting entities:', error);
      return false;
    }
  }
}
