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
/**
 * Retrieval Augmented Generation (RAG) implementation using Geode and LangChain.
 * Uses GeodeService for vector similarity search and document storage.
 * Uses NVIDIA API for LLM inference.
 *
 * Architecture:
 * - Uses Geode DocumentChunk nodes for Pure RAG (stores full text chunks with embeddings)
 * - Separate from Entity nodes used for knowledge graph entities
 * - Documents are retrieved via vector cosine similarity and fed to LLM for answer generation
 */

import { ChatOpenAI } from "@langchain/openai";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { EmbeddingsService } from './embeddings';
import { GeodeService } from './geode';

export class RAGService {
  private static instance: RAGService;
  private geodeService: GeodeService;
  private embeddingsService: EmbeddingsService;
  private llm: ChatOpenAI | null = null;
  private initialized: boolean = false;
  private isInitializing: boolean = false;

  private constructor() {
    this.geodeService = GeodeService.getInstance();
    this.embeddingsService = EmbeddingsService.getInstance();
  }

  /**
   * Get the singleton instance of RAGService
   */
  public static getInstance(): RAGService {
    if (!RAGService.instance) {
      RAGService.instance = new RAGService();
    }
    return RAGService.instance;
  }

  /**
   * Initialize the RAG service
   */
  public async initialize(): Promise<void> {
    if (this.initialized || this.isInitializing) {
      return;
    }

    this.isInitializing = true;

    try {
      console.log('Initializing RAG service...');

      // Prefer a local vLLM deployment when configured, falling back to
      // NVIDIA's cloud API. Mirrors the pattern in lib/llm-service.ts and
      // lib/langchain-service.ts's getVllmModel().
      const vllmBaseUrl = process.env.VLLM_BASE_URL;

      if (vllmBaseUrl) {
        // Remote vLLM endpoints may require Bearer auth (VLLM_API_KEY);
        // local vLLM ignores the placeholder key.
        this.llm = new ChatOpenAI({
          modelName: process.env.VLLM_MODEL || 'meta-llama/Llama-3.2-3B-Instruct',
          temperature: 0.2,
          maxTokens: 1024,
          apiKey: process.env.VLLM_API_KEY || 'vllm',
          configuration: {
            baseURL: vllmBaseUrl,
            timeout: 120000,
          },
          modelKwargs: {
            top_p: 0.95,
            frequency_penalty: 0,
            presence_penalty: 0
          }
        });

        console.log(`vLLM LLM initialized successfully (baseURL=${vllmBaseUrl})`);
      } else {
        // Check for NVIDIA API key
        const nvidiaApiKey = process.env.NVIDIA_API_KEY;
        if (!nvidiaApiKey) {
          const error = new Error('Either VLLM_BASE_URL or NVIDIA_API_KEY is required for RAG service. Please set one of these environment variables.');
          console.error('RAG Initialization Error:', error.message);
          throw error;
        }

        // Initialize NVIDIA LLM using ChatOpenAI with NVIDIA's base URL
        this.llm = new ChatOpenAI({
          modelName: "nvidia/llama-3.3-nemotron-super-49b-v1.5",
          temperature: 0.2,
          maxTokens: 1024,
          apiKey: nvidiaApiKey,
          configuration: {
            baseURL: "https://integrate.api.nvidia.com/v1",
            timeout: 120000,
          },
          modelKwargs: {
            top_p: 0.95,
            frequency_penalty: 0,
            presence_penalty: 0
          }
        });

        console.log('NVIDIA LLM initialized successfully');
      }

      // Initialize Geode and embeddings
      await this.geodeService.initialize();
      await this.embeddingsService.initialize();

      this.initialized = true;
      console.log('RAG service initialized successfully');
    } catch (error) {
      console.error('Error initializing RAG service:', error);
      this.isInitializing = false;
      throw error;
    } finally {
      this.isInitializing = false;
    }
  }

  /**
   * Store documents in Geode for retrieval
   * @param documents Array of text documents to store
   * @param metadata Optional metadata for the documents
   */
  public async storeDocuments(
    documents: string[],
    metadata?: Record<string, any>[]
  ): Promise<void> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (!documents || documents.length === 0) {
      console.warn('No documents provided to store');
      return;
    }

    console.log(`Storing ${documents.length} documents in Geode`);

    // Generate embeddings via EmbeddingsService
    const embeddings = await this.embeddingsService.encode(documents);

    // Store document chunks in Geode
    await this.geodeService.storeDocumentChunks(documents, embeddings, metadata);
    console.log(`Successfully stored ${documents.length} document embeddings in Geode`);
  }

  /**
   * Perform question answering with document retrieval using RAG
   * @param query User query
   * @param topK Number of most similar documents to retrieve
   * @returns Object containing the answer and document count
   */
  public async retrievalQA(query: string, topK: number = 5): Promise<{ answer: string; documentCount: number }> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (!this.llm) {
      throw new Error('LLM not initialized');
    }

    console.log(`Performing RAG query with topK=${topK}`);

    // Generate embedding for the query
    const queryEmbeddings = await this.embeddingsService.encode([query]);
    const queryEmbedding = queryEmbeddings[0];

    // Find similar documents using Geode vector search
    const similarDocs = await this.geodeService.findSimilarDocuments(queryEmbedding, topK);

    if (!similarDocs || similarDocs.length === 0) {
      console.log('No relevant documents found, falling back to LLM general knowledge');

      const fallbackPromptTemplate = PromptTemplate.fromTemplate(`
You are a helpful assistant answering questions based on your general knowledge.
Since no specific information was found in the knowledge base, please provide the best answer you can.

Question: {query}

Answer:
`);

      const fallbackChain = RunnableSequence.from([
        {
          query: () => query,
        },
        fallbackPromptTemplate,
        this.llm,
        new StringOutputParser(),
      ]);

      const answer = await fallbackChain.invoke({});
      return {
        answer: `[Note: No specific information was found in the knowledge base. This answer is based on general knowledge.]\n\n${answer}`,
        documentCount: 0
      };
    }

    console.log(`Found ${similarDocs.length} relevant document chunks`);

    // Extract text from retrieved documents
    const context = similarDocs
      .map((doc) => doc.metadata?.text || '')
      .filter((text) => text.length > 0)
      .join('\n\n');

    console.log(`Extracted context length: ${context.length} characters`);

    if (!context || context.trim().length === 0) {
      console.log('Retrieved documents have no content, falling back to LLM');
      const fallbackPromptTemplate = PromptTemplate.fromTemplate(`
You are a helpful assistant answering questions based on your general knowledge.

Question: {query}

Answer:
`);

      const fallbackChain = RunnableSequence.from([
        {
          query: () => query,
        },
        fallbackPromptTemplate,
        this.llm,
        new StringOutputParser(),
      ]);

      const answer = await fallbackChain.invoke({});
      return {
        answer: `[Note: No specific information was found in the knowledge base. This answer is based on general knowledge.]\n\n${answer}`,
        documentCount: similarDocs.length
      };
    }

    // Define prompt template for RAG
    const promptTemplate = PromptTemplate.fromTemplate(`
Answer the question based only on the following context from the knowledge base.
If you cannot find the answer in the context, say "I cannot find this information in the knowledge base."

Context:
{context}

Question: {query}

Answer:
`);

    // Create retrieval chain using RunnableSequence
    const retrievalChain = RunnableSequence.from([
      {
        context: () => context,
        query: () => query,
      },
      promptTemplate,
      this.llm,
      new StringOutputParser(),
    ]);

    console.log('Generating answer with NVIDIA LLM...');

    try {
      const answer = await retrievalChain.invoke({});
      console.log('RAG query completed successfully');
      console.log(`Answer length: ${answer.length} characters`);
      console.log(`Retrieved ${similarDocs.length} document chunks`);
      return {
        answer,
        documentCount: similarDocs.length
      };
    } catch (error) {
      console.error('Error generating answer with NVIDIA LLM:', error);
      throw error;
    }
  }

  /**
   * Perform similar document retrieval without QA
   * @param query Query text
   * @param topK Number of documents to retrieve
   * @returns Array of retrieved documents with similarity scores
   */
  public async retrieveSimilarDocuments(
    query: string,
    topK: number = 5
  ): Promise<Array<{ text: string; score: number; metadata?: any }>> {
    if (!this.initialized) {
      await this.initialize();
    }

    // Generate embedding for the query
    const queryEmbeddings = await this.embeddingsService.encode([query]);
    const queryEmbedding = queryEmbeddings[0];

    // Find similar documents using Geode vector search
    const results = await this.geodeService.findSimilarDocuments(queryEmbedding, topK);

    return results.map((doc) => ({
      text: doc.metadata?.text || '',
      score: doc.score,
      metadata: doc.metadata
    }));
  }
}

export default RAGService.getInstance();
