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
import { NextResponse } from 'next/server';
import { GeodeService } from '@/lib/geode';

/**
 * Get vector database stats from Geode
 */
export async function GET(): Promise<NextResponse> {
  try {
    const geode = GeodeService.getInstance();
    await geode.initialize();

    const stats = await geode.getStats();

    return NextResponse.json({
      ...stats,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error getting vector stats:', error);

    let errorMessage = error instanceof Error ? error.message : String(error);

    return NextResponse.json(
      {
        error: `Failed to get vector stats: ${errorMessage}`,
        totalVectorCount: 0,
        source: 'error',
        httpHealthy: false,
        timestamp: new Date().toISOString()
      },
      { status: 200 }
    );
  }
}
