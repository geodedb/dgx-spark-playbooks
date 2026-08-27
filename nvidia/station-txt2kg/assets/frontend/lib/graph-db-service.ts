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
import { GeodeService } from './geode';

/**
 * GraphDBService provides a unified interface for graph database operations.
 * All operations delegate to GeodeService.
 */
export class GraphDBService {
  private static instance: GraphDBService;

  private constructor() {}

  public static getInstance(): GraphDBService {
    if (!GraphDBService.instance) {
      GraphDBService.instance = new GraphDBService();
    }
    return GraphDBService.instance;
  }

  public async initialize(url?: string): Promise<void> {
    await GeodeService.getInstance().initialize(url);
  }

  public isInitialized(): boolean {
    return GeodeService.getInstance().isInitialized();
  }

  public async importTriples(
    triples: { subject: string; predicate: string; object: string }[]
  ): Promise<void> {
    await GeodeService.getInstance().importTriples(triples);
  }

  public async getGraphData(): Promise<{
    nodes: Array<{
      id: string;
      labels: string[];
      [key: string]: any;
    }>;
    relationships: Array<{
      id: string;
      source: string;
      target: string;
      type: string;
      [key: string]: any;
    }>;
  }> {
    return await GeodeService.getInstance().getGraphData();
  }

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
    await GeodeService.getInstance().logQuery(query, queryMode, metrics);
  }

  public async getQueryLogs(limit: number = 100): Promise<any[]> {
    return await GeodeService.getInstance().getQueryLogs(limit);
  }

  public async close(): Promise<void> {
    await GeodeService.getInstance().close();
  }

  public getDriverInfo(): Record<string, any> {
    return GeodeService.getInstance().getDriverInfo();
  }

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
    return await GeodeService.getInstance().graphTraversal(keywords, maxDepth, maxResults);
  }

  public async clearDatabase(): Promise<void> {
    await GeodeService.getInstance().clearDatabase();
  }
}
