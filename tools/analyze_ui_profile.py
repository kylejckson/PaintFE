"""Summarise opt-in local UI traces. Idle frame gaps are excluded.

These are CPU callback and frame-delivery timings, not GPU completion/presentation
measurements. Run: python tools/analyze_ui_profile.py path/to/ui-profile.jsonl
"""
import json
import sys
from pathlib import Path


def percentiles(values):
    values = sorted(values)
    if not values:
        return None
    return {name: round(values[min(len(values)-1, int((len(values)-1)*q))], 3)
            for name, q in (("median", .5), ("p95", .95), ("p99", .99))}


def analyze(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    rows = []
    for line in lines[1:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A writer may have a final, incomplete record while running.
    active = [row for i, row in enumerate(rows)
              if i > 0 and row.get("active") and rows[i-1].get("active")]
    return {
        "samples": len(rows), "active_samples": len(active),
        "active_frame_gap_ms": percentiles([r["gap_ms"] for r in active]),
        "previous_cpu_ms": percentiles([r["previous_cpu_ms"] for r in active
                                         if r.get("previous_cpu_ms") is not None]),
        "stages_ms": {stage: percentiles([r["ms"][i] for r in active])
                      for i, stage in enumerate(header["stages"])},
    }


if __name__ == "__main__":
    print(json.dumps(analyze(sys.argv[1]), indent=2))
