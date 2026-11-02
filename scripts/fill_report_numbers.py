#!/usr/bin/env python3
"""Fill the Run D placeholders in docs/REPORT_ADDITIONS.md and docs/PROJECT_EXPLAINED.md
from data/evaluation/eval_results_local.json and the exact-match timings in logs/evaluation.log."""

import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
summary = json.loads((ROOT / "data/evaluation/eval_results_local.json").read_text())["summary"]

KEYS = [  # (report id, guide id, summary key, digits)
    ("G9", "validity", "citation_validity_mean", 2),
    ("G10", "coverage", "cited_claim_coverage_mean", 2),
    ("G11", "support", "citation_support_score_mean", 2),
    ("G12", "halluc", "hallucination_rate", 2),
    ("G13", "response", "response_rate", 2),
    ("G14", "retprec", "retrieval_precision_mean", 2),
    ("G15", "confidence", "average_confidence", 2),
    ("G16", "latency", "mean_latency_seconds", 1),
]
values = {}
for gid, did, key, digits in KEYS:
    v = summary.get(key)
    values[gid] = values[did] = "n/a" if v is None else f"{v:.{digits}f}"

# Exact-match scan time: from "Query classified" to "Hybrid search returned" for
# queries that have an identifier (semantic weight 0.3); includes the dense search.
stamp = lambda line: datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f")
scans, start = [], None
for line in (ROOT / "logs/evaluation.log").read_text().splitlines():
    if "Query classified as" in line:
        start = stamp(line) if "semantic=0.3" in line else None
    elif "Hybrid search returned" in line and start:
        scans.append((stamp(line) - start).total_seconds())
        start = None
values["N3"] = f"{statistics.mean(scans):.0f}" if scans else "n/a"

for doc in ("docs/REPORT_ADDITIONS.md", "docs/PROJECT_EXPLAINED.md"):
    p = ROOT / doc
    t = p.read_text()
    t = re.sub(r"\[TBD-(G\d+|N3)\]", lambda m: values.get(m.group(1), m.group(0)), t)
    t = re.sub(r"\[TBD-D-([a-z]+)\]", lambda m: values.get(m.group(1), m.group(0)), t)
    p.write_text(t)
print(json.dumps(values, indent=1), f"\nexact-match scans timed: {len(scans)}", file=sys.stderr)
