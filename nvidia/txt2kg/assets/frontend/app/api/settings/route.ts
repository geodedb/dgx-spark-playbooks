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

// In-memory storage for settings
let serverSettings: Record<string, string> = {};
let settingsInitialized = false;

function ensureSettingsInitialized(): void {
  if (!settingsInitialized) {
    serverSettings = {
      graph_db_type: 'geode',
      geode_url: process.env.GEODE_URL || '',
    };
    settingsInitialized = true;
    console.log('[SETTINGS] Initialized at runtime with graph_db_type: "geode"');
  }
}

/**
 * API Route to sync client settings with server environment variables
 */
export async function POST(request: NextRequest): Promise<NextResponse> {
  try {
    ensureSettingsInitialized();

    const { settings } = await request.json();

    if (!settings || typeof settings !== 'object') {
      return NextResponse.json({ error: 'Settings object is required' }, { status: 400 });
    }

    // Merge but always enforce geode as the graph_db_type
    serverSettings = { ...serverSettings, ...settings, graph_db_type: 'geode' };

    return NextResponse.json({
      success: true,
      message: 'Settings updated successfully'
    });
  } catch (error) {
    console.error('Error updating settings:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: errorMessage }, { status: 500 });
  }
}

/**
 * GET /api/settings
 * Retrieve settings from the server side
 */
export async function GET(request: NextRequest): Promise<NextResponse> {
  try {
    ensureSettingsInitialized();

    const url = new URL(request.url);
    const key = url.searchParams.get('key');

    if (key) {
      return NextResponse.json({
        [key]: serverSettings[key] || null
      });
    }

    return NextResponse.json({
      settings: serverSettings
    });
  } catch (error) {
    console.error('Error retrieving settings:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: errorMessage }, { status: 500 });
  }
}

/**
 * Helper function to get a setting value (for use in other API routes)
 */
export function getSetting(key: string): string | null {
  ensureSettingsInitialized();
  return serverSettings[key] || null;
}

/**
 * Get the currently selected graph database type -- always 'geode'
 */
export function getGraphDbType(): 'geode' {
  return 'geode';
}
