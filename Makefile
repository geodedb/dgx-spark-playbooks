# DGX Spark Playbooks - Makefile
# Run individual playbooks with: make <playbook-name>
# Stop with: make <playbook-name>-down
# View logs with: make <playbook-name>-logs

SHELL := /bin/bash
ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
NVIDIA := $(ROOT)nvidia

.PHONY: help list

help: ## Show this help
	@echo "DGX Spark Playbooks"
	@echo "==================="
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}'

list: ## List all playbook directories
	@ls -1 $(NVIDIA)/

# ==============================================================================
# Docker Compose playbooks (up/down/logs pattern)
# ==============================================================================

# --- txt2kg -------------------------------------------------------------------
# NOTE: start.sh uses -f which skips the override file containing the aarch64
# health check fix. We call docker compose directly with the override included.

TXT2KG_COMPOSE := $(NVIDIA)/txt2kg/assets/deploy/compose

.PHONY: txt2kg txt2kg-vllm txt2kg-down txt2kg-down-vllm txt2kg-logs

txt2kg: ## Start txt2kg (Geode + Ollama)
	cd $(TXT2KG_COMPOSE) && docker compose up -d

txt2kg-vllm: ## Start txt2kg with vLLM (GPU-accelerated)
	cd $(TXT2KG_COMPOSE) && docker compose -f docker-compose.vllm.yml -f docker-compose.vllm.override.yml up -d

txt2kg-down: ## Stop txt2kg
	cd $(TXT2KG_COMPOSE) && docker compose down

txt2kg-down-vllm: ## Stop txt2kg (vLLM variant)
	cd $(TXT2KG_COMPOSE) && docker compose -f docker-compose.vllm.yml -f docker-compose.vllm.override.yml down

txt2kg-logs: ## Follow txt2kg logs
	cd $(TXT2KG_COMPOSE) && docker compose logs -f

# --- multi-agent-chatbot ------------------------------------------------------

CHATBOT_DIR := $(NVIDIA)/multi-agent-chatbot/assets

.PHONY: multi-agent-chatbot multi-agent-chatbot-models multi-agent-chatbot-down \
        multi-agent-chatbot-down-models multi-agent-chatbot-logs multi-agent-chatbot-download

multi-agent-chatbot-download: ## Download models for multi-agent-chatbot
	cd $(CHATBOT_DIR) && chmod +x model_download.sh && ./model_download.sh

multi-agent-chatbot: ## Start multi-agent-chatbot (app + Geode)
	cd $(CHATBOT_DIR) && docker compose up -d --build

multi-agent-chatbot-models: ## Start multi-agent-chatbot model servers
	cd $(CHATBOT_DIR) && docker compose -f docker-compose-models.yml up -d --build

multi-agent-chatbot-all: ## Start multi-agent-chatbot app + models together
	cd $(CHATBOT_DIR) && docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build

multi-agent-chatbot-down: ## Stop multi-agent-chatbot app
	cd $(CHATBOT_DIR) && docker compose down

multi-agent-chatbot-down-models: ## Stop multi-agent-chatbot model servers
	cd $(CHATBOT_DIR) && docker compose -f docker-compose-models.yml down

multi-agent-chatbot-down-all: ## Stop all multi-agent-chatbot services
	cd $(CHATBOT_DIR) && docker compose down && docker compose -f docker-compose-models.yml down

multi-agent-chatbot-logs: ## Follow multi-agent-chatbot logs
	cd $(CHATBOT_DIR) && docker compose logs -f

# --- pytorch-fine-tune --------------------------------------------------------

.PHONY: pytorch-fine-tune pytorch-fine-tune-down pytorch-fine-tune-logs

pytorch-fine-tune: ## Start PyTorch fine-tuning (Docker Swarm)
	cd $(NVIDIA)/pytorch-fine-tune/assets && docker compose up -d

pytorch-fine-tune-down: ## Stop PyTorch fine-tuning
	cd $(NVIDIA)/pytorch-fine-tune/assets && docker compose down

pytorch-fine-tune-logs: ## Follow PyTorch fine-tuning logs
	cd $(NVIDIA)/pytorch-fine-tune/assets && docker compose logs -f

# ==============================================================================
# Docker run playbooks (run/stop pattern)
# ==============================================================================

# --- jax ----------------------------------------------------------------------

.PHONY: jax jax-build jax-stop

jax-build: ## Build JAX container
	cd $(NVIDIA)/jax/assets && docker build -t jax-on-spark .

jax: jax-build ## Run JAX marimo notebook (port 8080)
	docker run --gpus all --rm -it -p 8080:8080 --name jax-on-spark jax-on-spark

jax-stop: ## Stop JAX container
	-docker stop jax-on-spark

# --- open-webui ---------------------------------------------------------------

.PHONY: open-webui open-webui-stop

open-webui: ## Run Open WebUI (port 8080)
	docker run -d -p 8080:8080 --gpus=all \
		-v open-webui:/app/backend/data \
		-v open-webui-ollama:/root/.ollama \
		--name open-webui \
		ghcr.io/open-webui/open-webui:ollama

open-webui-stop: ## Stop Open WebUI
	-docker stop open-webui && docker rm open-webui

# --- nim-llm ------------------------------------------------------------------

.PHONY: nim-llm nim-llm-stop

nim-llm: ## Run NIM LLM (port 8000, requires NGC_API_KEY)
	@test -n "$(NGC_API_KEY)" || (echo "Error: NGC_API_KEY not set" && exit 1)
	docker run -it --rm --name nim-llm-demo --gpus all \
		--shm-size=16GB \
		-e NGC_API_KEY=$(NGC_API_KEY) \
		-v "$(HOME)/.cache/nim:/opt/nim/.storage" \
		-p 8000:8000 \
		nvcr.io/nim/meta/llama-3.1-8b-instruct-dgx-spark:latest

nim-llm-stop: ## Stop NIM LLM container
	-docker stop nim-llm-demo

# --- sglang -------------------------------------------------------------------

.PHONY: sglang sglang-stop

sglang: ## Run SGLang server (port 30000, set MODEL= to override)
	docker run --gpus all --ipc=host --rm -d \
		--name sglang \
		-p 30000:30000 \
		lmsysorg/sglang:spark \
		python -m sglang.launch_server \
		--model-path $(or $(MODEL),meta-llama/Llama-3.1-8B-Instruct) \
		--port 30000 --host 0.0.0.0

sglang-stop: ## Stop SGLang container
	-docker stop sglang

# --- vllm ---------------------------------------------------------------------

.PHONY: vllm vllm-stop

vllm: ## Run vLLM server (port 8000, set MODEL= to override)
	docker run --gpus all --ipc=host --rm -d \
		--name vllm \
		-p 8000:8000 \
		lmsysorg/vllm:spark \
		python -m vllm.entrypoints.openai.api_server \
		--model $(or $(MODEL),meta-llama/Llama-3.1-8B-Instruct) \
		--port 8000 --host 0.0.0.0

vllm-stop: ## Stop vLLM container
	-docker stop vllm

# ==============================================================================
# Script-based playbooks
# ==============================================================================

# --- portfolio-optimization ---------------------------------------------------

.PHONY: portfolio-optimization

portfolio-optimization: ## Run portfolio optimization Jupyter notebook (port 8888)
	cd $(NVIDIA)/portfolio-optimization/assets && bash setup/start_playbook.sh

# --- single-cell --------------------------------------------------------------

.PHONY: single-cell

single-cell: ## Run single-cell analysis Jupyter notebook (port 8888)
	cd $(NVIDIA)/single-cell/assets && bash setup/start_playbook.sh

# --- flux-finetuning ----------------------------------------------------------

FLUX_DIR := $(NVIDIA)/flux-finetuning/assets

.PHONY: flux-download flux-inference flux-inference-build flux-train flux-train-build

flux-download: ## Download FLUX models
	cd $(FLUX_DIR) && sh download.sh

flux-inference-build: ## Build FLUX inference (ComfyUI) image
	cd $(FLUX_DIR) && docker build -f Dockerfile.inference -t flux-comfyui .

flux-inference: flux-inference-build ## Run FLUX ComfyUI inference
	cd $(FLUX_DIR) && sh launch_comfyui.sh

flux-train-build: ## Build FLUX training image
	cd $(FLUX_DIR) && docker build -f Dockerfile.train -t flux-train .

flux-train: flux-train-build ## Run FLUX fine-tuning training
	cd $(FLUX_DIR) && sh launch_train.sh

# ==============================================================================
# System service / install playbooks
# ==============================================================================

.PHONY: ollama ollama-pull

ollama: ## Install Ollama
	curl -fsSL https://ollama.com/install.sh | sh

ollama-pull: ## Pull a model (set MODEL=, default: qwen2.5:32b)
	ollama pull $(or $(MODEL),qwen2.5:32b)

.PHONY: tailscale

tailscale: ## Install Tailscale
	curl -fsSL https://tailscale.com/install.sh | sh
	@echo "Run 'sudo tailscale up' to connect"

.PHONY: live-vlm-webui live-vlm-webui-run

live-vlm-webui: ## Install live-vlm-webui
	pip install live-vlm-webui

live-vlm-webui-run: ## Run live-vlm-webui (port 8090)
	live-vlm-webui

.PHONY: lm-studio

lm-studio: ## Install LM Studio (llmster)
	curl -fsSL https://lmstudio.ai/install.sh | bash

# ==============================================================================
# Docker container playbooks (interactive)
# ==============================================================================

.PHONY: llama-factory unsloth nemo-fine-tune cuda-x-data-science

llama-factory: ## Run LLaMA Factory container (interactive)
	docker run --gpus all -it --rm \
		--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
		-v $(HOME)/.cache/huggingface:/root/.cache/huggingface \
		--name llama-factory \
		nvcr.io/nvidia/pytorch:25.11-py3

unsloth: ## Run Unsloth container (interactive)
	docker run --gpus all -it --rm \
		--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
		-v $(HOME)/.cache/huggingface:/root/.cache/huggingface \
		-v $(NVIDIA)/unsloth/assets:/workspace \
		--name unsloth \
		nvcr.io/nvidia/pytorch:25.11-py3

nemo-fine-tune: ## Run NeMo fine-tuning container (interactive)
	docker run --gpus all -it --rm \
		--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
		-v $(HOME)/.cache/huggingface:/root/.cache/huggingface \
		--name nemo-fine-tune \
		nvcr.io/nvidia/pytorch:25.11-py3

multi-modal-inference: ## Run multi-modal inference container (interactive)
	docker run --gpus all -it --rm \
		--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
		-v $(HOME)/.cache/huggingface:/root/.cache/huggingface \
		--name multi-modal-inference \
		nvcr.io/nvidia/pytorch:25.11-py3

# --- nvfp4-quantization -------------------------------------------------------

.PHONY: nvfp4-quantization nvfp4-serve

nvfp4-quantization: ## Run NVFP4 quantization container (interactive, set HF_TOKEN=)
	docker run --rm -it --gpus all \
		--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
		-e HF_TOKEN=$(HF_TOKEN) \
		-v $(HOME)/output_models:/workspace/model \
		--name nvfp4 \
		nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev

nvfp4-serve: ## Serve NVFP4 quantized model (port 8000)
	docker run --rm -d --gpus all \
		--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
		-v $(HOME)/output_models:/workspace/model \
		-p 8000:8000 \
		--name nvfp4-serve \
		nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
		trtllm-serve /workspace/model --backend pytorch --port 8000

nvfp4-serve-stop: ## Stop NVFP4 serve container
	-docker stop nvfp4-serve

# --- speculative-decoding -----------------------------------------------------

.PHONY: speculative-decoding

speculative-decoding: ## Run speculative decoding container (interactive, set HF_TOKEN=)
	docker run --rm -it --gpus all \
		--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
		-e HF_TOKEN=$(HF_TOKEN) \
		-p 8355:8355 \
		--name speculative-decoding \
		nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6

# --- trt-llm ------------------------------------------------------------------

.PHONY: trt-llm trt-llm-stop

trt-llm: ## Run TensorRT-LLM container (interactive, set HF_TOKEN=)
	docker run --rm -it --gpus all \
		--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
		-e HF_TOKEN=$(HF_TOKEN) \
		-p 8355:8355 \
		--name trt-llm \
		nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6

trt-llm-stop: ## Stop TensorRT-LLM container
	-docker stop trt-llm

# ==============================================================================
# Utility targets
# ==============================================================================

.PHONY: status stop-all

status: ## Show running playbook containers
	@echo "Running containers:"
	@docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "  Docker not available"
	@echo ""
	@echo "Docker Compose projects:"
	@docker compose ls 2>/dev/null || echo "  None"
