#!/usr/bin/env python3
"""Side-by-side A/B report for two run_eval.py result files.

Usage: compare.py <label_a> <label_b>   (reads results/<label>.json)
Prints a per-prompt + aggregate comparison: throughput, output length,
tool-call success, refusal behavior, and truncated outputs for eyeballing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RES = Path(__file__).parent / "results"


def load(label: str) -> dict:
    return json.loads((RES / f"{label}.json").read_text())


def idx(d: dict) -> dict:
    return {r["id"]: r for r in d["results"]}


def short(s: str, n: int = 160) -> str:
    return " ".join((s or "").split())[:n]


def main() -> None:
    a_label, b_label = sys.argv[1], sys.argv[2]
    a, b = load(a_label), load(b_label)
    ai, bi = idx(a), idx(b)

    print(f"# A/B comparison\n")
    print(f"- **A** = `{a_label}` ({a['model']}) — avg {a['avg_tok_per_s']} tok/s, ok {a['n_ok']}/{a['n']}")
    print(f"- **B** = `{b_label}` ({b['model']}) — avg {b['avg_tok_per_s']} tok/s, ok {b['n_ok']}/{b['n']}")
    spd = (b['avg_tok_per_s'] / a['avg_tok_per_s']) if a['avg_tok_per_s'] and b['avg_tok_per_s'] else None
    if spd:
        print(f"- throughput B/A = **{spd:.2f}×** ({'faster' if spd>=1 else 'slower'})")
    print()

    print("| prompt | category | A tok/s | B tok/s | A ok | B ok | A tool | B tool |")
    print("|---|---|--:|--:|:-:|:-:|:-:|:-:|")
    for pid in ai:
        ra, rb = ai[pid], bi.get(pid, {})
        print(f"| {pid} | {ra.get('category','')} | {ra.get('tok_per_s','-')} | {rb.get('tok_per_s','-')} "
              f"| {'Y' if ra.get('ok') else 'N'} | {'Y' if rb.get('ok') else 'N'} "
              f"| {'Y' if ra.get('tool_calls') else '-'} | {'Y' if rb.get('tool_calls') else '-'} |")

    print("\n## Side-by-side outputs (truncated)\n")
    for pid in ai:
        ra, rb = ai[pid], bi.get(pid, {})
        print(f"### {pid}  ({ra.get('category','')})")
        print(f"- **A:** {short(ra.get('content',''))}")
        if ra.get("tool_calls"):
            fc = ra["tool_calls"][0].get("function", {})
            print(f"  - tool: `{fc.get('name')}({fc.get('arguments')})`")
        print(f"- **B:** {short(rb.get('content',''))}")
        if rb.get("tool_calls"):
            fc = rb["tool_calls"][0].get("function", {})
            print(f"  - tool: `{fc.get('name')}({fc.get('arguments')})`")
        print()


if __name__ == "__main__":
    main()
