#!/usr/bin/env python3
"""Model-agnostic A/B eval runner for the spark vLLM OpenAI-compatible endpoint.

Runs a fixed prompt set against /v1/chat/completions, recording outputs and
performance metrics so two models (e.g. Qwen3.6 vs MiniMax-M3-REAP50) can be
compared on identical inputs. Greedy decoding (temp 0) for reproducibility.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


def load_prompts(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def detect_model(client: httpx.Client, base_url: str) -> str:
    r = client.get(f"{base_url}/v1/models", timeout=10)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def run_one(client: httpx.Client, base_url: str, model: str, p: dict,
            temperature: float, max_tokens: int, req_timeout: float) -> dict:
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": p["user"]}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if "tools" in p:
        body["tools"] = p["tools"]
        body["tool_choice"] = "auto"
    t0 = time.time()
    try:
        r = client.post(f"{base_url}/v1/chat/completions", json=body, timeout=req_timeout)
        dt = time.time() - t0
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        usage = data.get("usage", {})
        ct = usage.get("completion_tokens") or 0
        return {
            "id": p["id"], "category": p["category"], "ok": True,
            "content": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or "",
            "tool_calls": msg.get("tool_calls") or [],
            "finish_reason": data["choices"][0].get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": ct,
            "latency_s": round(dt, 2),
            "tok_per_s": round(ct / dt, 1) if dt > 0 and ct else None,
        }
    except Exception as e:  # noqa: BLE001 — capture per-prompt, keep going
        return {"id": p["id"], "category": p["category"], "ok": False,
                "error": f"{type(e).__name__}: {e}", "latency_s": round(time.time() - t0, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://curumo:8000")
    ap.add_argument("--label", required=True, help="output label, e.g. qwen3.6-baseline")
    ap.add_argument("--prompts", default=str(Path(__file__).parent / "prompts.jsonl"))
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--req-timeout", type=float, default=150.0)
    args = ap.parse_args()

    prompts = load_prompts(Path(args.prompts))
    progress = Path(args.out_dir) / f"{args.label}.progress.txt"
    progress.parent.mkdir(parents=True, exist_ok=True)
    progress.write_text("")
    with httpx.Client() as client:
        model = detect_model(client, args.base_url)
        print(f"[{args.label}] endpoint model = {model}; {len(prompts)} prompts", flush=True)
        results = []
        for p in prompts:
            res = run_one(client, args.base_url, model, p, args.temperature, args.max_tokens, args.req_timeout)
            flag = "ok " if res["ok"] else "ERR"
            extra = f"{res.get('completion_tokens','?')}tok {res.get('tok_per_s','?')}tok/s" if res["ok"] else res.get("error", "")
            line = f"  [{flag}] {res['id']:<22} {res.get('latency_s','?')}s {extra}"
            print(line, flush=True)
            with progress.open("a") as fh:
                fh.write(line + "\n")
            results.append(res)

    ok = [r for r in results if r["ok"]]
    agg = {
        "label": args.label, "model": model, "base_url": args.base_url,
        "temperature": args.temperature, "n": len(results), "n_ok": len(ok),
        "avg_tok_per_s": round(sum(r["tok_per_s"] for r in ok if r["tok_per_s"]) / max(1, sum(1 for r in ok if r["tok_per_s"])), 1) if ok else None,
        "total_completion_tokens": sum(r.get("completion_tokens") or 0 for r in ok),
        "results": results,
    }
    out = Path(args.out_dir) / f"{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(agg, indent=2, ensure_ascii=False))
    print(f"[{args.label}] wrote {out}  (ok {len(ok)}/{len(results)}, avg {agg['avg_tok_per_s']} tok/s)")


if __name__ == "__main__":
    main()
