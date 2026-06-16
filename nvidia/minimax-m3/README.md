# MiniMax-M3 (REAP50 NVFP4) — 2× DGX Spark serving

Reproducible, **Ansible-driven** deployment of `sparkarena/Minimax-M3-v0-NVFP4-REAP50`
across the `olorin` + `curumo` GB10 DGX Spark pair via **sparkrun/SGLang**
(TP=2, MiniMax Sparse Attention, NVFP4), with **systemd boot persistence**.

Unlike the docker-compose playbooks in this repo, this topic is implemented as an
Ansible role (the cluster work — NCCL tuning, systemd units, GPU-clock pin — is
multi-host and stateful). The Makefile targets wrap `ansible-playbook`.

## What you'll accomplish

- Serve MiniMax-M3 (REAP50) on `http://10.0.0.26:8000/v1` (OpenAI-compatible).
- Persist it across reboots (`spark-minimax-m3.service` on the head; GPU-clock pin on both nodes).
- Everything reproducible: `make minimax-m3`.

## Prerequisites

- Both Sparks reachable over the `10.0.0.0/24` mgmt fabric; passwordless sudo for the run user.
- `uv`/`uvx`, Docker, and the `scitrera/dgx-spark-sglang-mm:v0` image (the role pulls it).
- ~130 GB free per node for the model (`make minimax-m3-stage` downloads + fans out).
- Ansible + collections (`community.docker`, `community.general`, `ansible.posix`) on the control host.

## Instructions

```bash
make minimax-m3-stage     # one-time: download model on head, rsync to worker
make minimax-m3           # deploy units (idempotent)
make minimax-m3-start     # start serving
make minimax-m3-logs      # follow the SGLang serve log
make minimax-m3-down      # stop
```

Direct: `cd nvidia/minimax-m3/ansible && ansible-playbook site.yml`. See
[`ansible/README.md`](ansible/README.md) for tags and variables.

## Validated facts (live, 2026-06-16)

| Item | Value |
|---|---|
| Context ceiling | **512K** (`max_model_len=524288`). SGLang derives 512K from the model's rope config and **refuses 1M** ("incorrect outputs/CUDA errors") — needle-recall validated at 170K & 338K. |
| NCCL (GB10 multi-node) | single RoCE HCA `rocep1s0f0` (10.177.0.x DAC) + `NCCL_IB_GID_INDEX=3` + `enP7s7` bootstrap + `NCCL_NET_PLUGIN=none`. Inactive `roceP2p*` HCAs or `NCCL_IB_DISABLE=1` both break init. |
| Throughput | ~18 tok/s (vs Qwen3.6-35B-AWQ ~39). |
| Model download | authenticated **plain HTTPS** (`HF_HUB_DISABLE_XET=1`); Xet hangs, unauthenticated throttles. |
| GPU-clock pin | `nvidia-smi -lgc 200,2150` (firmware-shutdown guard under load). |

## A/B evaluation

`eval/` holds the model-agnostic harness used to compare REAP50 vs the prior
Qwen3.6 deployment (`run_eval.py`, `compare.py`, `longctx_probe.py`) plus the
recorded baselines under `eval/results/`.

## Rollback

The previous `spark-vllm-cluster.service` (Qwen3.6-35B-AWQ) and its weights remain
on the cluster: `systemctl stop spark-minimax-m3 && systemctl start spark-vllm-cluster`.
