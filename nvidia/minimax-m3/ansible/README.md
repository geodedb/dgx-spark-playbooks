# MiniMax-M3 (REAP50 NVFP4) — reproducible DGX Spark deployment

Serves `sparkarena/Minimax-M3-v0-NVFP4-REAP50` across the **olorin + curumo** GB10
DGX Spark pair via **sparkrun/SGLang** (TP=2, MSA, NVFP4), with boot persistence.

## What it deploys

| Component | Where | Purpose |
|---|---|---|
| `spark-gpu-clock.service` | both nodes | Pins GB10 clocks to `200,2150` MHz (firmware-shutdown guard) |
| `/etc/minimax-m3/recipe.yaml` | both nodes | sparkrun/SGLang recipe (validated NCCL env + context len) |
| `spark-m3-launch` wrapper | head | start/stop wrapper around `sparkrun run/stop` |
| `spark-minimax-m3.service` | head | Runs `sparkrun run` on boot; launches both-node containers |
| ufw allows + ssh host-key trust | both / head | inter-node NCCL + sparkrun's internal ssh |

## Usage

```bash
cd ansible
ansible-playbook site.yml --syntax-check          # validate
ansible-playbook site.yml --check                 # dry run
ansible-playbook site.yml                          # deploy units (assumes model staged)
ansible-playbook site.yml --tags stage             # ALSO download (head) + rsync (workers) the 121GB model
ansible-playbook site.yml --tags start             # start the service now
```

After deploy: `systemctl status spark-minimax-m3` on the head; endpoint at
`http://10.0.0.26:8000/v1/models`. Roll back to Qwen with the original
`spark-vllm-cluster.service` (its weights remain cached).

## Validated facts (live, 2026-06-16)

- **Context limit: 524288 (512K)** is this build's safe max. SGLang derives 512K
  from the model's rope config and **refuses 1,048,576** ("incorrect outputs /
  CUDA errors") unless `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` is forced —
  so 512K is the ceiling. Needle-recall validated at ~170K and ~338K tokens.
- **NCCL on GB10 multi-node** only works with: single persistent RoCE HCA
  `rocep1s0f0` (the 10.177.0.x DAC) + `NCCL_IB_GID_INDEX=3` + `enP7s7` socket
  bootstrap + `NCCL_NET_PLUGIN=none`. Listing the inactive `roceP2p*` HCAs or
  disabling IB both break it. (Mined from the working FSDP/vLLM stacks.)
- **Throughput** ~18 tok/s (vs Qwen3.6-35B-AWQ ~39 tok/s). Cross-node copy of the
  121GB model goes over the 10G mgmt fabric (~500 MB/s); the manual 200G DAC IP
  was non-persistent, but NCCL still uses the DAC via RoCE.
- HF download must be **authenticated + plain HTTPS** (`HF_HUB_DISABLE_XET=1`);
  Xet hangs and unauthenticated throttles on this egress.

## Notes / follow-ups

- Runs as user `{{ m3_run_user = benji }}` (owns the HF cache, sparkrun ssh mesh,
  uvx). Change `m3_run_user` in `group_vars/all.yml` to use a service account
  (requires re-staging the cache + ssh mesh under that user).
- The 200G CX-7 DAC (`enp1s0f0np0`, 10.177.0.x) is already persistent via the
  existing netplan; this playbook does not manage it. See NVIDIA's
  `connect-two-sparks/assets/cx7-netplan.yaml` if it ever needs re-creation.
