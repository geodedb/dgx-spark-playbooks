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
 * API endpoint for disconnecting from Geode
 * POST /api/graph-db/disconnect
 */
export async function POST(request: NextRequest): Promise<NextResponse> {
  try {
    const geode = GeodeService.getInstance();
    console.log('Disconnecting from geode...');

    if (geode.isInitialized()) {
      await geode.close();
      return NextResponse.json({
        success: true,
        message: 'Successfully disconnected from geode',
        type: 'geode'
      });
    }

    return NextResponse.json({
      success: false,
      message: 'No active geode connection to disconnect',
      type: 'geode'
    });
  } catch (error) {
    console.error('Error disconnecting from graph database:', error);
    return NextResponse.json(
      {
        error: `Failed to disconnect from graph database: ${error instanceof Error ? error.message : String(error)}`,
        type: 'geode'
      },
      { status: 500 }
    );
  }
}
