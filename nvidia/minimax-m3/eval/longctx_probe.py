#!/usr/bin/env python3
"""Long-context needle-in-haystack probe for validating a context-limit bump.

Builds a prompt of ~target_tokens, embeds an ACCESS CODE at ~75% depth, asks the
model to recall it. Reports actual prompt_tokens, recall correctness, and timing
(prefill is the long pole at high context). Usage:
    longctx_probe.py <approx_tokens> [base_url]
"""
from __future__ import annotations

import sys
import time

import httpx

approx_tokens = int(sys.argv[1])
base_url = sys.argv[2] if len(sys.argv) > 2 else "http://curumo:8000"
CODE = "ZEPHYR-7741-QX"

# ~4 chars/token; build varied filler lines so it isn't trivially compressible.
line = "Log entry {i}: routine telemetry sample, subsystem nominal, checksum ok, no action required.\n"
approx_chars = approx_tokens * 4
filler, n, chars = [], 0, 0
while chars < approx_chars:
    s = line.format(i=n)
    filler.append(s)
    chars += len(s)
    n += 1
# insert the needle at ~75% depth
needle_at = int(len(filler) * 0.75)
filler[needle_at] = f"Log entry {needle_at}: CRITICAL ACCESS CODE = {CODE} (store securely).\n"
haystack = "".join(filler)

prompt = (
    "The following log contains exactly one ACCESS CODE somewhere inside it.\n\n"
    + haystack
    + "\n\nWhat is the ACCESS CODE? Reply with only the code, nothing else."
)

with httpx.Client() as c:
    model = c.get(f"{base_url}/v1/models", timeout=10).json()["data"][0]["id"]
    t0 = time.time()
    r = c.post(f"{base_url}/v1/chat/completions", timeout=1200, json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": 640,
    })
    dt = time.time() - t0
    r.raise_for_status()
    d = r.json()
    msg = d["choices"][0]["message"]
    out = (msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")
    pt = d.get("usage", {}).get("prompt_tokens")
    ok = CODE in out
    print(f"target~{approx_tokens} | actual_prompt_tokens={pt} | recall={'PASS' if ok else 'FAIL'} "
          f"| latency={dt:.1f}s | answer={(msg.get('content') or '')[:40]!r}")
    sys.exit(0 if ok else 2)
