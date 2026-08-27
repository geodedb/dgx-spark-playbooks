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
 * Ensure the Geode connection is ready, optionally using a URL from the request.
 */
async function ensureConnection(request?: NextRequest): Promise<void> {
  const geode = GeodeService.getInstance();

  if (geode.isInitialized()) {
    return;
  }

  const url = request?.nextUrl.searchParams.get('url') || undefined;
  await geode.initialize(url);
}

/**
 * GET handler for retrieving graph data from Geode
 */
export async function GET(request: NextRequest): Promise<NextResponse> {
  console.log('[graph-db GET] Request received');
  try {
    console.log('[graph-db GET] Ensuring connection...');
    await ensureConnection(request);

    const geode = GeodeService.getInstance();
    console.log('[graph-db GET] Using database type: geode');

    console.log('[graph-db GET] Fetching graph data...');
    const graphData = await geode.getGraphData();
    console.log(`[graph-db GET] Got ${graphData.nodes.length} nodes, ${graphData.relationships.length} relationships`);

    const nodes = graphData.nodes.map(node => ({
      ...node,
      name: node.name || `Node ${node.id}`,
      label: node.labels?.[0] || 'Entity',
      val: 1,
      color: node.labels?.includes('Entity') ? '#ff6b6b' : '#4ecdc4'
    }));

    const links = graphData.relationships.map(rel => ({
      ...rel,
      label: rel.type || 'RELATED_TO'
    }));

    const connectionUrl = request.nextUrl.searchParams.get('url')
      || process.env.GEODE_URL
      || 'Not specified';

    return NextResponse.json({
      nodes,
      links,
      connectionUrl,
      databaseType: 'geode'
    });
  } catch (error) {
    console.error('Error in graph database GET handler:', error);
    return NextResponse.json(
      { error: `Failed to fetch graph data: ${error instanceof Error ? error.message : String(error)}` },
      { status: 500 }
    );
  }
}

/**
 * POST handler for importing triples into Geode
 */
export async function POST(request: NextRequest): Promise<NextResponse> {
  try {
    await ensureConnection(request);

    const geode = GeodeService.getInstance();
    const body = await request.json();

    if (!body.triples || !Array.isArray(body.triples)) {
      return NextResponse.json(
        { error: 'Invalid request: triples array is required' },
        { status: 400 }
      );
    }

    await geode.importTriples(body.triples);

    return NextResponse.json({
      success: true,
      message: `Successfully imported ${body.triples.length} triples into geode`,
      databaseType: 'geode'
    });
  } catch (error) {
    console.error('Error in graph database POST handler:', error);
    return NextResponse.json(
      { error: `Failed to import triples: ${error instanceof Error ? error.message : String(error)}` },
      { status: 500 }
    );
  }
}
