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
 * API endpoint to create a test query log
 * This is for debugging purposes only
 */
export async function GET(request: NextRequest): Promise<NextResponse> {
  try {
    console.log('[Test] Creating test query log');

    const geode = GeodeService.getInstance();
    await geode.initialize();

    const query = request.nextUrl.searchParams.get('query') || 'Test query for debugging';
    const queryMode = (request.nextUrl.searchParams.get('mode') || 'traditional') as 'traditional' | 'vector-search' | 'pure-rag';
    const executionTime = parseInt(request.nextUrl.searchParams.get('time') || '300');
    const resultCount = parseInt(request.nextUrl.searchParams.get('count') || '5');

    console.log(`[Test] Adding test query: "${query}" (${queryMode})`);

    await geode.logQuery(
      query,
      queryMode,
      {
        executionTimeMs: executionTime,
        relevanceScore: 0,
        precision: 0,
        recall: 0,
        resultCount: resultCount
      }
    );

    const logs = await geode.getQueryLogs(10);

    return NextResponse.json({
      success: true,
      message: `Test query "${query}" added successfully`,
      logs: logs.slice(0, 3)
    });
  } catch (error) {
    console.error('[Test] Error creating test query log:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json(
      { error: errorMessage },
      { status: 500 }
    );
  }
}
