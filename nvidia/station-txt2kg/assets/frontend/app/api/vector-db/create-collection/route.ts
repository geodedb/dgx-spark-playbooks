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
 * Create collection API endpoint (no-op for Geode)
 * POST /api/vector-db/create-collection
 */
export async function POST(): Promise<NextResponse> {
  try {
    const geode = GeodeService.getInstance();
    await geode.initialize();
    await geode.createCollection();

    const stats = await geode.getStats();

    return NextResponse.json({
      success: true,
      message: 'Geode collection ready (no-op -- nodes are created dynamically)',
      httpHealthy: stats.httpHealthy || false
    });
  } catch (error) {
    console.error('Error in create-collection endpoint:', error);

    return NextResponse.json(
      {
        success: false,
        error: `Failed to initialize Geode: ${error instanceof Error ? error.message : String(error)}`
      },
      { status: 500 }
    );
  }
}
