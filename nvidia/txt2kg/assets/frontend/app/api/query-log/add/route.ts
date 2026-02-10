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

/**
 * Simple endpoint to directly add a query log entry
 */
export async function GET(request: NextRequest): Promise<NextResponse> {
  try {
    const query = request.nextUrl.searchParams.get('query') || 'How does machine learning work?';
    const count = parseInt(request.nextUrl.searchParams.get('count') || '20');

    const geode = GeodeService.getInstance();
    await geode.initialize();

    // Log the primary query
    await geode.logQuery(query, 'traditional', {
      executionTimeMs: 0,
      relevanceScore: 0,
      precision: 0,
      recall: 0,
      resultCount: 0
    });

    // Log additional queries when count is high enough
    if (count >= 10) {
      await geode.logQuery('What are the applications of artificial intelligence?', 'traditional', {
        executionTimeMs: 0,
        relevanceScore: 0,
        precision: 0,
        recall: 0,
        resultCount: 0
      });

      await geode.logQuery('Explain the principles of deep learning', 'traditional', {
        executionTimeMs: 0,
        relevanceScore: 0,
        precision: 0,
        recall: 0,
        resultCount: 0
      });
    }

    const logs = await geode.getQueryLogs(5);

    return NextResponse.json({
      success: true,
      message: `Added query log for "${query}" with count ${count}`,
      addedQuery: { query, count },
      logs
    });
  } catch (error) {
    console.error('Error adding query log:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : String(error)
    }, { status: 500 });
  }
}
