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
/**
 * Client-side initialization utilities
 * This file contains functions for initializing the application on the client side
 */

/**
 * Initialize default database settings if not already set
 * Called before syncing with server to ensure defaults are available
 */
export function initializeDefaultSettings() {
  if (typeof window === 'undefined') {
    return; // Only run on client side
  }

  // Set default Geode connection URL
  if (!localStorage.getItem('geode_url')) {
    localStorage.setItem('geode_url', 'quic://localhost:3141');
  }
}

/**
 * Synchronize settings from localStorage with the server
 * Called on app initialization to ensure server has access to client settings
 */
export async function syncSettingsWithServer() {
  if (typeof window === 'undefined') {
    return; // Only run on client side
  }

  // Initialize default settings first
  initializeDefaultSettings();

  // Collect all relevant settings from localStorage
  const settings: Record<string, string> = {};

  // NVIDIA API settings
  const nvidiaEmbeddingsModel = localStorage.getItem('nvidia_embeddings_model');
  if (nvidiaEmbeddingsModel) {
    settings.nvidia_embeddings_model = nvidiaEmbeddingsModel;
  }

  const embeddingsProvider = localStorage.getItem('embeddings_provider');
  if (embeddingsProvider) {
    settings.embeddings_provider = embeddingsProvider;
  }

  // Geode database settings
  const geodeUrl = localStorage.getItem('geode_url');
  if (geodeUrl) {
    settings.geode_url = geodeUrl;
  }

  // NVIDIA Nemotron API key
  const nvidiaApiKey = localStorage.getItem('NVIDIA_API_KEY');
  if (nvidiaApiKey) {
    settings.NVIDIA_API_KEY = nvidiaApiKey;
  }

  
  // Skip the API call if there are no settings to sync
  if (Object.keys(settings).length === 0) {
    return;
  }
  
  // Send settings to server
  try {
    const response = await fetch('/api/settings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ settings }),
    });
    
    if (!response.ok) {
      throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
    }
    
    console.log('Client settings synchronized with server');
  } catch (error) {
    console.error('Failed to sync settings with server:', error);
  }
} 