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
 * Load triples from Geode for use with the TXT2KG dataset
 * @param url Geode connection URL
 * @returns Array of triples as strings in the format "subject predicate object"
 */
export async function loadTriplesFromGeode(url?: string): Promise<string[]> {
  try {
    const geodeService = GeodeService.getInstance();

    if (!geodeService.isInitialized()) {
      await geodeService.initialize(url);
    }

    const query = `MATCH (s:Entity)-[r:RELATIONSHIP]->(o:Entity) RETURN s.name + ' ' + r.type + ' ' + o.name AS triple`;

    // Use graphTraversal-style approach: query via the client
    // GeodeService exposes queryAll indirectly; use getGraphData + format
    const graphData = await geodeService.getGraphData();

    const triples: string[] = [];
    const nodeIdToName = new Map<string, string>();
    for (const node of graphData.nodes) {
      nodeIdToName.set(node.id, node.name);
    }

    for (const rel of graphData.relationships) {
      const subject = nodeIdToName.get(rel.source);
      const object = nodeIdToName.get(rel.target);
      if (subject && object) {
        triples.push(`${subject} ${rel.type} ${object}`);
      }
    }

    console.log(`Loaded ${triples.length} triples from Geode`);
    return triples;
  } catch (error) {
    console.error('Error loading triples from Geode:', error);
    throw error;
  }
}
