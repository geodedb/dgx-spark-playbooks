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
#
# Author: William Benton <wbenton@nvidia.com> 


FROM nvidia/cuda:13.3.0-runtime-ubuntu24.04

RUN ldconfig
COPY --from=ghcr.io/astral-sh/uv:latest	/uv /uvx /bin/

RUN mkdir /app
WORKDIR /app

RUN uv init --python 3.12 && uv venv && uv pip install --no-cache "marimo==0.16.5" && uv pip install --no-cache "jax[cuda13]==0.7.2" && uv pip install --no-cache "numpy==2.3.3" && uv pip install --no-cache "plotly==6.3.0" && uv pip install --no-cache "opencv-python-headless==4.12.0.88" && uv pip install --no-cache "tqdm==4.67.1"

COPY *.py *.mp4 /app/

# No curl/wget/python on PATH in this CUDA runtime base image; probe the
# marimo server's TCP listener directly via bash's /dev/tcp pseudo-device.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD bash -c 'exec 3<>/dev/tcp/127.0.0.1/8080 || exit 1'

CMD ["sh", "-c", "uv run marimo edit 00-toc.py --host 0.0.0.0 --port 8080 --headless --no-token --no-sandbox"]
