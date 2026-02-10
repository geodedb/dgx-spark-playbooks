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
import { NextRequest, NextResponse } from 'next/server';
import { GeodeService } from '@/lib/geode';
import remoteBackendInstance from '@/lib/remote-backend';
import RAGService from '@/lib/rag';
import queryLoggerService, { QueryLogSummary } from '@/lib/query-logger';

/**
 * Metrics API that provides performance statistics about the RAG system
 */
export async function GET(request: NextRequest): Promise<NextResponse> {
  try {
    const geode = GeodeService.getInstance();
    await geode.initialize();

    // Get graph stats from Geode
    const graphData = await geode.getGraphData();

    const uniqueEntities = new Set<string>();
    graphData.nodes.forEach((node: any) => uniqueEntities.add(node.name));

    const totalTriples = graphData.relationships.length;

    // Get vector stats from Geode
    let vectorStats = {
      totalVectors: 0,
      avgQueryTime: 0,
      avgRelevanceScore: 0
    };

    try {
      const stats = await geode.getStats();
      vectorStats = {
        totalVectors: stats.totalVectorCount || 0,
        avgQueryTime: stats.averageQueryTime || 0,
        avgRelevanceScore: stats.averageRelevanceScore || 0
      };
    } catch (error) {
      console.warn('Could not fetch vector stats from Geode:', error);
    }

    // Get real query logs instead of mock data
    let queryLogs: QueryLogSummary[] = [];
    let precision = 0;
    let recall = 0;
    let f1Score = 0;
    let avgQueryTime = vectorStats.avgQueryTime || 0;
    let avgRelevance = 0;
    let queryTimesByMode: Record<string, number> = {};

    try {
      if (!queryLoggerService.isInitialized()) {
        await queryLoggerService.initialize();
      }

      console.log('Getting query logs from file');
      queryLogs = await queryLoggerService.getQueryLogs(25);
      console.log(`Found ${queryLogs.length} query logs from file-based logger`);

      if (queryLogs.length > 0) {
        const logsWithMetrics = queryLogs.filter(log =>
          log.metrics.avgPrecision > 0 ||
          log.metrics.avgRecall > 0 ||
          log.metrics.avgExecutionTimeMs > 0
        );

        const logsWithRelevance = queryLogs.filter(log => log.metrics.avgRelevanceScore > 0);

        if (logsWithMetrics.length > 0) {
          precision = logsWithMetrics.reduce((sum, log) => sum + (log.metrics.avgPrecision || 0), 0) / logsWithMetrics.length;
          recall = logsWithMetrics.reduce((sum, log) => sum + (log.metrics.avgRecall || 0), 0) / logsWithMetrics.length;
          avgQueryTime = logsWithMetrics.reduce((sum, log) => sum + (log.metrics.avgExecutionTimeMs || 0), 0) / logsWithMetrics.length;
          f1Score = precision > 0 && recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
        }

        if (logsWithRelevance.length > 0) {
          avgRelevance = logsWithRelevance.reduce((sum, log) => sum + (log.metrics.avgRelevanceScore || 0), 0) / logsWithRelevance.length;
        }
      }

      const logsByMode = queryLogs.reduce((acc, log) => {
        const mode = log.queryMode || 'traditional';
        if (!acc[mode]) acc[mode] = [];
        acc[mode].push(log);
        return acc;
      }, {} as Record<string, typeof queryLogs>);

      Object.entries(logsByMode).forEach(([mode, logs]) => {
        const logsWithTime = logs.filter(log => log.metrics.avgExecutionTimeMs > 0);
        if (logsWithTime.length > 0) {
          queryTimesByMode[mode] = logsWithTime.reduce((sum, log) =>
            sum + log.metrics.avgExecutionTimeMs, 0
          ) / logsWithTime.length;
        }
      });
    } catch (error) {
      console.warn('Error getting query logs from file:', error);
    }

    const topQueries = queryLogs.length > 0
      ? queryLogs
          .sort((a, b) => b.count - a.count)
          .slice(0, 5)
          .map(log => ({
            query: log.query,
            count: log.count
          }))
      : [];

    const metrics = {
      totalTriples,
      totalEntities: uniqueEntities.size,
      avgQueryTime,
      avgRelevance: avgRelevance || vectorStats.avgRelevanceScore || 0,
      precision,
      recall,
      f1Score,
      topQueries,
      queryTimesByMode: queryTimesByMode || {},
      queryLogStats: {
        totalQueryLogs: queryLogs.length,
        totalExecutions: queryLogs.reduce((sum, log) => sum + log.executionCount, 0),
        lastQueriedAt: queryLogs.length > 0 ? queryLogs[0].lastQueried : null
      }
    };

    return NextResponse.json(metrics);
  } catch (error) {
    console.error('Error fetching metrics:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: errorMessage }, { status: 500 });
  }
}
