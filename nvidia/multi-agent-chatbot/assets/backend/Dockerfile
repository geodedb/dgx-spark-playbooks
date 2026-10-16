#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
FROM python:3.15

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl=8.14.1-2+deb13u4 && \
    rm -rf /var/lib/apt/lists/* && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

RUN . $HOME/.local/bin/env
ENV PATH="/root/.local/bin:$PATH"

# uv installs into /root/.local/bin, which is not reachable by a non-root
# user (home dirs default to 0700). Copy the binaries somewhere on the
# default PATH so they still work once we drop root below.
RUN cp /root/.local/bin/uv /root/.local/bin/uvx /usr/local/bin/

COPY pyproject.toml ./
COPY . .

RUN uv sync

# Drop root privileges for the running process.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
ENV HOME=/home/appuser
USER appuser

# Matches the docker-compose healthcheck already defined for this service.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8000/available_models

CMD ["uv", "run", "--", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]