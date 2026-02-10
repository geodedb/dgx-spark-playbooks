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
import type { Triple } from '@/types/graph';

/**
 * API endpoint for fetching all triples from Geode
 * GET /api/graph-db/triples
 */
export async function GET(req: NextRequest): Promise<NextResponse> {
  try {
    const geode = GeodeService.getInstance();
    await geode.initialize();

    console.log('Fetching all triples from geode...');

    const graphData = await geode.getGraphData();

    const nodeMap = new Map<string, string>();
    for (const node of graphData.nodes) {
      nodeMap.set(node.id, node.name);
    }

    const triples: Triple[] = [];
    for (const rel of graphData.relationships) {
      const subject = nodeMap.get(rel.source);
      const object = nodeMap.get(rel.target);
      const predicate = rel.type;

      if (subject && predicate && object) {
        triples.push({ subject, predicate, object });
      }
    }

    const uniqueTriples = deduplicateTriples(triples);
    console.log(`Successfully fetched ${uniqueTriples.length} unique triples from geode`);

    return NextResponse.json({
      success: true,
      triples: uniqueTriples,
      count: uniqueTriples.length,
      databaseType: 'geode'
    });
  } catch (error) {
    console.error('Error fetching triples from graph database:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json(
      { error: `Failed to fetch triples: ${errorMessage}` },
      { status: 500 }
    );
  }
}

/**
 * Deduplicate triples by lowercased subject|predicate|object key
 */
function deduplicateTriples(triples: Triple[]): Triple[] {
  const seen = new Set<string>();
  return triples.filter(triple => {
    const key = `${triple.subject.toLowerCase()}|${triple.predicate.toLowerCase()}|${triple.object.toLowerCase()}`;
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

/**
 * API endpoint for storing triples in Geode
 * POST /api/graph-db/triples
 */
export async function POST(req: NextRequest): Promise<NextResponse> {
  try {
    const body = await req.json();
    const { triples, documentName } = body;

    if (!triples || !Array.isArray(triples)) {
      return NextResponse.json({ error: 'Triples are required' }, { status: 400 });
    }

    const geode = GeodeService.getInstance();
    await geode.initialize();

    console.log(`Storing ${triples.length} triples in geode from document "${documentName || 'unnamed'}"`);

    const validTriples = triples.filter((triple: any) => {
      return (
        triple &&
        typeof triple.subject === 'string' && triple.subject.trim() !== '' &&
        typeof triple.predicate === 'string' && triple.predicate.trim() !== '' &&
        typeof triple.object === 'string' && triple.object.trim() !== ''
      );
    }) as Triple[];

    console.log(`Found ${validTriples.length} valid triples to store`);

    await geode.importTriples(validTriples);

    if (documentName) {
      try {
        await geode.markDocumentAsProcessed(documentName, validTriples.length);
      } catch (error) {
        console.error('Error marking document as processed:', error);
      }
    }

    return NextResponse.json({
      success: true,
      message: 'Triples stored successfully in geode',
      count: validTriples.length,
      documentName,
      databaseType: 'geode'
    });
  } catch (error) {
    console.error('Error storing triples in graph database:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json(
      { error: `Failed to store triples: ${errorMessage}` },
      { status: 500 }
    );
  }
}
