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
 * Get the default graph database type. Always returns 'geode'.
 */
export function getDefaultGraphDbType(): string {
  return 'geode';
}

/**
 * Get the graph database service instance.
 */
export function getGraphDbService(): GeodeService {
  return GeodeService.getInstance();
}

/**
 * Initialize the graph database connection via GeodeService.
 */
export async function initializeGraphDb(): Promise<void> {
  await GeodeService.getInstance().initialize(process.env.GEODE_URL);
}
