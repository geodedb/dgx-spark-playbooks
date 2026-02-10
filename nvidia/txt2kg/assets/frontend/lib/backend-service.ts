//
// SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import axios from 'axios';
import { GraphDBService } from './graph-db-service';
import { GeodeService } from './geode';
import type { Triple } from '@/types/graph';

/**
 * Backend service that combines graph database for storage and Geode for embeddings.
 *
 * Two distinct modes:
 * 1. Knowledge Graph Mode: Stores triples in graph DB + entity embeddings in Geode
 * 2. Pure RAG Mode: Stores document chunks as DocumentChunk nodes in Geode (via RAGService)
 *
 * Use processTriples() for knowledge graph ingestion
 * Use storeDocumentChunks() for Pure RAG document ingestion
 */
export class BackendService {
  private graphDBService: GraphDBService;
  private geodeService: GeodeService;
  private sentenceTransformerUrl: string = 'http://sentence-transformers:80';
  private modelName: string = 'all-MiniLM-L6-v2';
  private static instance: BackendService;
  private initialized: boolean = false;

  private constructor() {
    this.graphDBService = GraphDBService.getInstance();
    this.geodeService = GeodeService.getInstance();

    if (process.env.SENTENCE_TRANSFORMER_URL) {
      this.sentenceTransformerUrl = process.env.SENTENCE_TRANSFORMER_URL;
    }
    if (process.env.MODEL_NAME) {
      this.modelName = process.env.MODEL_NAME;
    }
  }

  /**
   * Get the singleton instance of BackendService
   */
  public static getInstance(): BackendService {
    if (!BackendService.instance) {
      BackendService.instance = new BackendService();
    }
    return BackendService.instance;
  }

  /**
   * Initialize the backend services
   */
  public async initialize(): Promise<void> {
    // Initialize Geode (handles both graph and vector operations)
    if (!this.geodeService.isInitialized()) {
      try {
        await this.geodeService.initialize(process.env.GEODE_URL);
        console.log('Geode initialized successfully in backend service');
      } catch (error) {
        console.error('Failed to initialize Geode in backend service:', error);
        if (process.env.NODE_ENV === 'development') {
          console.log('Development mode: Continuing despite Geode initialization error');
        } else {
          throw new Error('Geode service initialization failed');
        }
      }
    }

    // Check if sentence-transformer service is available
    try {
      const response = await axios.get(`${this.sentenceTransformerUrl}/health`);
      console.log(`Connected to SentenceTransformer service: ${response.data.model}`);
      this.initialized = true;
    } catch (error) {
      console.error(`Failed to connect to sentence-transformer service: ${error}`);
      if (process.env.NODE_ENV === 'development') {
        console.log('Development mode: Continuing despite sentence transformer error');
        this.initialized = true;
      } else {
        throw new Error('Sentence transformer service is not available');
      }
    }
  }

  /**
   * Check if the backend is initialized
   */
  public get isInitialized(): boolean {
    return this.initialized && this.geodeService.isInitialized();
  }

  /**
   * Generate embeddings using the sentence-transformer service
   */
  private async generateEmbeddings(texts: string[]): Promise<number[][]> {
    try {
      const response = await axios.post(`${this.sentenceTransformerUrl}/embed`, {
        texts,
        batch_size: 32
      });

      return response.data.embeddings;
    } catch (error) {
      console.error(`Error generating embeddings: ${error}`);
      throw new Error('Failed to generate embeddings');
    }
  }

  /**
   * Convert our triple format to database format
   */
  private convertTriples(triples: Triple[]): { subject: string; predicate: string; object: string }[] {
    return triples.map(triple => ({
      subject: triple.subject,
      predicate: triple.predicate,
      object: triple.object
    }));
  }

  /**
   * Process and store triples in graph database and embeddings in Geode
   */
  public async processTriples(triples: Triple[]): Promise<void> {
    // Preprocess triples: lowercase and remove duplicates
    const processedTriples = triples.map(triple => ({
      subject: triple.subject.toLowerCase(),
      predicate: triple.predicate.toLowerCase(),
      object: triple.object.toLowerCase()
    }));

    // Remove duplicate triples
    const uniqueTriples = Array.from(
      new Map(processedTriples.map(triple => [JSON.stringify(triple), triple])).values()
    );

    console.log(`Processed ${triples.length} triples, removed ${triples.length - uniqueTriples.length} duplicates`);

    // Store triples in Geode graph
    console.log('Storing triples in Geode database');
    await this.graphDBService.importTriples(this.convertTriples(uniqueTriples));

    // Extract unique entities from triples
    const entities = new Set<string>();
    for (const triple of uniqueTriples) {
      entities.add(triple.subject);
      entities.add(triple.object);
    }

    // Generate embeddings for entities in batches
    const entityList = Array.from(entities);
    const batchSize = 256;
    const entityEmbeddings = new Map<string, number[]>();
    const textContent = new Map<string, string>();

    console.log(`Generating embeddings for ${entityList.length} entities`);

    for (let i = 0; i < entityList.length; i += batchSize) {
      const batch = entityList.slice(i, i + batchSize);
      console.log(`Processing batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(entityList.length/batchSize)}`);

      const embeddings = await this.generateEmbeddings(batch);

      for (let j = 0; j < batch.length; j++) {
        entityEmbeddings.set(batch[j], embeddings[j]);
        textContent.set(batch[j], batch[j]);
      }
    }

    // Store embeddings in Geode
    await this.geodeService.storeEmbeddings(entityEmbeddings, textContent);

    console.log(`Backend processing complete: ${uniqueTriples.length} triples and ${entityList.length} entities stored using Geode`);
  }

  /**
   * Perform a traditional query using direct pattern matching on the graph
   */
  public async queryTraditional(queryText: string): Promise<Triple[]> {
    console.log(`Performing traditional graph query: "${queryText}"`);

    const keywords = this.extractKeywords(queryText);
    console.log(`Extracted keywords: ${keywords.join(', ')}`);

    const filteredKeywords = keywords.filter(kw => !this.isStopWord(kw));

    // Use Geode's native graph traversal
    console.log(`Using Geode graph traversal for keywords: ${filteredKeywords.join(', ')}`);

    try {
      const results = await this.graphDBService.graphTraversal(filteredKeywords, 2, 100);
      console.log(`Geode graph traversal found ${results.length} relevant triples`);

      console.log('Top 10 triples by confidence:');
      results.slice(0, 10).forEach((triple, idx) => {
        console.log(`  ${idx + 1}. [${triple.confidence.toFixed(3)}] ${triple.subject} -> ${triple.predicate} -> ${triple.object} (depth: ${triple.depth})`);
      });

      return results;
    } catch (error) {
      console.error('Error using Geode graph traversal, falling back to keyword-based search:', error);
    }

    // Fallback to keyword-based search
    console.log('Using fallback keyword-based search');

    const graphData = await this.graphDBService.getGraphData();
    console.log(`Retrieved graph from Geode with ${graphData.nodes.length} nodes and ${graphData.relationships.length} relationships`);

    const nodeIdToName = new Map<string, string>();
    for (const node of graphData.nodes) {
      nodeIdToName.set(node.id, node.name);
    }

    const matchingNodeIds = new Set<string>();
    for (const node of graphData.nodes) {
      for (const keyword of filteredKeywords) {
        if (node.name.toLowerCase().includes(keyword.toLowerCase())) {
          matchingNodeIds.add(node.id);
          break;
        }
      }
    }

    console.log(`Found ${matchingNodeIds.size} nodes matching keywords directly`);

    const relevantTriples: Triple[] = [];

    for (const rel of graphData.relationships) {
      const isSourceMatching = matchingNodeIds.has(rel.source);
      const isTargetMatching = matchingNodeIds.has(rel.target);

      if (isSourceMatching || isTargetMatching) {
        const sourceName = nodeIdToName.get(rel.source);
        const targetName = nodeIdToName.get(rel.target);

        if (sourceName && targetName) {
          let matchesRelationship = false;
          for (const keyword of filteredKeywords) {
            if (rel.type.toLowerCase().includes(keyword.toLowerCase())) {
              matchesRelationship = true;
              break;
            }
          }

          const relevance = (isSourceMatching ? 1 : 0) +
                           (isTargetMatching ? 1 : 0) +
                           (matchesRelationship ? 2 : 0);

          if (relevance > 0) {
            relevantTriples.push({
              subject: sourceName,
              predicate: rel.type,
              object: targetName,
              confidence: relevance / 4.0
            });
          }
        }
      }
    }

    relevantTriples.sort((a, b) =>
      (b.confidence || 0) - (a.confidence || 0)
    );

    console.log(`Found ${relevantTriples.length} relevant triples with traditional search`);
    return relevantTriples;
  }

  /**
   * Extract keywords from query text
   */
  private extractKeywords(text: string): string[] {
    return text.toLowerCase()
      .replace(/[.,?!;:()]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 2);
  }

  /**
   * Check if a word is a common stop word
   */
  private isStopWord(word: string): boolean {
    const stopWords = new Set([
      'the', 'and', 'are', 'for', 'was', 'with',
      'how', 'what', 'why', 'who', 'when', 'which',
      'many', 'much', 'from', 'have', 'has', 'had',
      'that', 'this', 'these', 'those', 'they', 'their'
    ]);
    return stopWords.has(word.toLowerCase());
  }

  /**
   * Query the backend for relevant information
   */
  public async query(
    queryText: string,
    kNeighbors: number = 4096,
    fanout: number = 400,
    numHops: number = 2,
    useTraditional: boolean = false
  ): Promise<Triple[]> {
    console.log(`Querying backend with Geode, useTraditional: ${useTraditional}`);

    if (useTraditional) {
      return this.queryTraditional(queryText);
    }

    // Generate embedding for query
    const queryEmbedding = (await this.generateEmbeddings([queryText]))[0];

    // Find nearest neighbors using Geode
    const seedNodes = await this.geodeService.findSimilarEntities(queryEmbedding, kNeighbors);
    console.log(`Found ${seedNodes.length} seed nodes for query: "${queryText}"`);

    // Get graph data from Geode
    const graphData = await this.graphDBService.getGraphData();
    console.log(`Retrieved graph from Geode with ${graphData.nodes.length} nodes and ${graphData.relationships.length} relationships`);

    // Build adjacency map for neighborhood exploration
    const adjacencyMap = new Map<string, string[]>();

    const nodeIdToName = new Map<string, string>();
    for (const node of graphData.nodes) {
      nodeIdToName.set(node.id, node.name);
      adjacencyMap.set(node.name, []);
    }

    for (const rel of graphData.relationships) {
      const sourceName = nodeIdToName.get(rel.source);
      const targetName = nodeIdToName.get(rel.target);

      if (sourceName && targetName) {
        const neighbors = adjacencyMap.get(sourceName) || [];
        neighbors.push(targetName);
        adjacencyMap.set(sourceName, neighbors);
      }
    }

    // Perform multi-hop exploration
    const visitedNodes = new Set<string>(seedNodes);
    const nodesToExplore = [...seedNodes];

    for (let hop = 0; hop < numHops; hop++) {
      const currentNodes = [...nodesToExplore];
      nodesToExplore.length = 0;

      for (const node of currentNodes) {
        const neighbors = adjacencyMap.get(node) || [];
        const limitedNeighbors = neighbors.slice(0, fanout);

        for (const neighbor of limitedNeighbors) {
          if (!visitedNodes.has(neighbor)) {
            visitedNodes.add(neighbor);
            nodesToExplore.push(neighbor);
          }
        }
      }

      console.log(`Hop ${hop+1}: Explored ${currentNodes.length} nodes, found ${nodesToExplore.length} new neighbors`);
    }

    // Extract relevant triples
    const relevantTriples: Triple[] = [];

    for (const rel of graphData.relationships) {
      const sourceName = nodeIdToName.get(rel.source);
      const targetName = nodeIdToName.get(rel.target);

      if (sourceName && targetName &&
         (visitedNodes.has(sourceName) || visitedNodes.has(targetName))) {
        relevantTriples.push({
          subject: sourceName,
          predicate: rel.type,
          object: targetName
        });
      }
    }

    const topK = 5;

    console.log(`Found ${relevantTriples.length} relevant triples, returning top ${topK * 5}`);
    return relevantTriples.slice(0, topK * 5);
  }

  /**
   * Query with LLM enhancement: retrieve triples and use LLM to generate answer
   */
  public async queryWithLLM(
    queryText: string,
    topK: number = 5,
    useTraditional: boolean = true,
    llmModel?: string,
    llmProvider?: string
  ): Promise<{ answer: string; triples: Triple[]; count: number }> {
    console.log(`Querying with LLM enhancement: "${queryText}", topK=${topK}, traditional=${useTraditional}`);

    // Step 1: Retrieve relevant triples using graph search
    const allTriples = await this.query(queryText, 4096, 400, 2, useTraditional);

    // Step 2: Take top K triples for context
    const topTriples = allTriples.slice(0, topK);
    console.log(`Using top ${topTriples.length} triples as context for LLM`);

    if (topTriples.length > 0) {
      console.log('First triple structure:', JSON.stringify(topTriples[0], null, 2));
    }

    if (topTriples.length === 0) {
      return {
        answer: "I couldn't find any relevant information in the knowledge graph to answer this question.",
        triples: [],
        count: 0
      };
    }

    // Step 3: Format triples as natural language context
    const context = topTriples
      .map(triple => {
        const predicate = triple.predicate
          .replace(/_/g, ' ')
          .replace(/-/g, ' ')
          .toLowerCase();
        return `${triple.subject} ${predicate} ${triple.object}`;
      })
      .join('. ');

    // Step 4: Use LLM to generate answer from context
    try {
      const prompt = `Answer the question based on the following context from the knowledge graph.

Context:
${context}

Question: ${queryText}

Answer:`;

      const finalProvider = llmProvider || 'ollama';
      const finalModel = llmModel || process.env.OLLAMA_MODEL || 'llama3.1:8b';

      console.log(`Using LLM: provider=${finalProvider}, model=${finalModel}`);

      let response;

      if (finalProvider === 'nvidia') {
        const nvidiaApiKey = process.env.NVIDIA_API_KEY;
        if (!nvidiaApiKey) {
          throw new Error('NVIDIA_API_KEY is required for NVIDIA provider. Please set the NVIDIA_API_KEY environment variable.');
        }

        const nvidiaUrl = 'https://integrate.api.nvidia.com/v1';

        response = await axios.post(`${nvidiaUrl}/chat/completions`, {
          model: finalModel,
          messages: [
            {
              role: 'system',
              content: '/think'
            },
            {
              role: 'user',
              content: prompt
            }
          ],
          temperature: 0.2,
          max_tokens: 4096,
          top_p: 0.95,
          frequency_penalty: 0,
          presence_penalty: 0,
          stream: false,
          min_thinking_tokens: 1024,
          max_thinking_tokens: 2048
        }, {
          headers: {
            'Authorization': `Bearer ${nvidiaApiKey}`,
            'Content-Type': 'application/json'
          },
          timeout: 120000
        });
      } else {
        const ollamaUrl = process.env.OLLAMA_BASE_URL || 'http://localhost:11434/v1';

        response = await axios.post(`${ollamaUrl}/chat/completions`, {
          model: finalModel,
          messages: [
            {
              role: 'system',
              content: 'You are a knowledgeable research assistant specializing in biomedical and scientific literature. Provide accurate, well-structured answers based on the provided context. Maintain a professional yet accessible tone, and clearly indicate when information is limited or uncertain.'
            },
            {
              role: 'user',
              content: prompt
            }
          ],
          temperature: 0.2,
          max_tokens: 800
        });
      }

      const messageData = response.data.choices[0].message;
      let answer = messageData.content || '';

      if (finalProvider === 'nvidia' && messageData.reasoning_content) {
        answer = `<think>\n${messageData.reasoning_content}\n</think>\n\n${answer}`;
        console.log('Formatted response with thinking content');
      }

      console.log('Returning triples (first one):', JSON.stringify(topTriples[0], null, 2));

      return {
        answer,
        triples: topTriples,
        count: topTriples.length
      };
    } catch (error) {
      console.error('Error calling LLM for answer generation:', error);
      return {
        answer: `Found ${topTriples.length} relevant triples:\n\n${context}`,
        triples: topTriples,
        count: topTriples.length
      };
    }
  }

  /**
   * Store document chunks for Pure RAG (separate from entity embeddings)
   */
  public async storeDocumentChunks(
    documents: string[],
    metadata?: Record<string, any>[]
  ): Promise<void> {
    console.log(`Storing ${documents.length} document chunks for Pure RAG`);

    const embeddings = await this.generateEmbeddings(documents);

    await this.geodeService.storeDocumentChunks(documents, embeddings, metadata);

    console.log(`Stored ${documents.length} document chunks in Geode`);
  }

  /**
   * Close connections to backend services
   */
  public async close(): Promise<void> {
    if (this.geodeService.isInitialized()) {
      await this.geodeService.close();
    }
    console.log('Backend service closed');
  }
}

export default BackendService.getInstance();
