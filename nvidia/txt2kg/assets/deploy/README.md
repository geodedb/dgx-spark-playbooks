# Deployment Configuration

This directory contains all deployment-related configuration for the txt2kg project.

## Structure

- **compose/**: Docker Compose configuration
  - `docker-compose.yml`: Geode + Ollama (default)
  - `docker-compose.vllm.yml`: Geode + vLLM (GPU-accelerated)

- **app/**: Frontend application Docker configuration
  - Dockerfile for Next.js application

- **services/**: Containerized services
  - **ollama/**: Ollama LLM inference service (default)
  - **vllm/**: vLLM inference service with GPU support (via `--vllm` flag)
  - **sentence-transformers/**: Sentence transformer service for embeddings
  - **gpu-viz/**: GPU-accelerated graph visualization services (run separately)
  - **gnn_model/**: Graph Neural Network model service (experimental)

## Usage

**Recommended: Use the start script**

```bash
# Default: Geode + Ollama
./start.sh

# Use Geode + vLLM (GPU-accelerated, for DGX Spark/GB300)
./start.sh --vllm

# Development mode (run frontend without Docker)
./start.sh --dev-frontend
```

**Manual Docker Compose commands:**

```bash
# Default: Geode + Ollama
docker compose -f deploy/compose/docker-compose.yml up -d

# Geode + vLLM
docker compose -f deploy/compose/docker-compose.vllm.yml up -d
```

## Services Included

### Default Stack (Geode + Ollama)
- **Next.js App**: Web UI on port 3001
- **Geode**: Graph + vector database on port 3141 (QUIC)
- **Ollama**: Local LLM inference on port 11434

### vLLM Stack (`--vllm` flag) - Geode + vLLM
- **Next.js App**: Web UI on port 3001
- **Geode**: Graph + vector database on port 3141 (QUIC)
- **vLLM**: GPU-accelerated LLM inference on port 8001

### Optional Services (run separately)
- **GPU-Viz Services**: See `services/gpu-viz/README.md` for GPU-accelerated visualization
- **GNN Model Service**: See `services/gnn_model/README.md` for experimental GNN-based RAG

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Default Stack (./start.sh)          │  vLLM Stack (--vllm)     │
├──────────────────────────────────────┼──────────────────────────┤
│                                      │                          │
│  ┌─────────────┐                     │  ┌─────────────┐         │
│  │   Next.js   │ port 3001           │  │   Next.js   │ 3001    │
│  └──────┬──────┘                     │  └──────┬──────┘         │
│         │                            │         │                │
│  ┌──────┴──────┐  ┌─────────────┐    │  ┌──────┴──────┐  ┌─────┐│
│  │   Geode     │  │   Ollama    │    │  │   Geode     │  │vLLM ││
│  │  port 3141  │  │ port 11434  │    │  │  port 3141  │  │8001 ││
│  │  (QUIC)     │  │             │    │  │  (QUIC)     │  │     ││
│  └─────────────┘  └─────────────┘    │  └─────────────┘  └─────┘│
│                                      │                          │
└──────────────────────────────────────┴──────────────────────────┘

Geode provides both graph and vector storage in a single database.
```
