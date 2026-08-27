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
import { EmbeddingsService } from '@/lib/embeddings';
import { GeodeService } from '@/lib/geode';

/**
 * Generate embeddings for text chunks and store them in Geode
 */
export async function POST(request: NextRequest): Promise<NextResponse> {
  try {
    const { documentId, content, documentName } = await request.json();

    if (!content) {
      return NextResponse.json(
        { error: 'Document content is required' },
        { status: 400 }
      );
    }

    const embeddingsService = EmbeddingsService.getInstance();
    console.log(`Using embeddings provider: ${process.env.EMBEDDINGS_PROVIDER || 'local'}`);

    const chunkSize = 200;
    const chunks = generateChunks(content, chunkSize);
    console.log(`Generated ${chunks.length} chunks from document`);

    const docPrefix = documentName
      ? documentName.replace(/[^a-zA-Z0-9]/g, '_').substring(0, 20)
      : documentId || 'doc';

    const chunkIds = chunks.map((_, index) => `${docPrefix}_chunk_${index}`);

    console.log('Generating embeddings for chunks...');
    const embeddings = await embeddingsService.encode(chunks);
    console.log(`Generated ${embeddings.length} embeddings`);

    const geode = GeodeService.getInstance();
    await geode.initialize();

    const entityEmbeddings = new Map<string, number[]>();
    const textContent = new Map<string, string>();

    for (let i = 0; i < chunkIds.length; i++) {
      entityEmbeddings.set(chunkIds[i], embeddings[i]);
      textContent.set(chunkIds[i], chunks[i]);
    }

    try {
      await geode.storeEmbeddings(entityEmbeddings, textContent);
    } catch (storeError) {
      console.error('Error storing embeddings in Geode:', storeError);
      return NextResponse.json(
        { error: `Failed to store embeddings: ${storeError instanceof Error ? storeError.message : String(storeError)}` },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: true,
      documentId: documentId || 'unnamed',
      chunks: chunks.length,
      embeddings: embeddings.length
    });
  } catch (error) {
    console.error('Error generating embeddings:', error);
    return NextResponse.json(
      { error: `Failed to generate embeddings: ${error instanceof Error ? error.message : String(error)}` },
      { status: 500 }
    );
  }
}

/**
 * Generate chunks from text content with sentence-aware splitting
 */
function generateChunks(content: string, chunkSize: number, overlap: number = 50): string[] {
  const chunks: string[] = [];
  const sentences = content.split(/(?<=[.!?])\s+/);

  let currentChunk = '';
  for (const sentence of sentences) {
    if (currentChunk.length + sentence.length > chunkSize && currentChunk.length > 0) {
      chunks.push(currentChunk.trim());
      const words = currentChunk.split(' ');
      currentChunk = words.slice(Math.max(0, words.length - overlap)).join(' ');
    }

    currentChunk += ' ' + sentence;
  }

  if (currentChunk.trim().length > 0) {
    chunks.push(currentChunk.trim());
  }

  return chunks;
}
