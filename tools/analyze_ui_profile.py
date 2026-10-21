"""Summarise opt-in UI JSONL traces. Timings are inclusive, never additive.

CPU callbacks and frame delivery do not measure GPU completion or presentation.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def stats(values):
    values = sorted(values)
    if not values:
        return None
    return {"n": len(values), **{
        key: round(values[min(len(values) - 1, int((len(values) - 1) * q))], 4)
        for key, q in [("p50", .5), ("p95", .95), ("p99", .99), ("max", 1)]
    }}


def analyse(path):
    lines = []
    raw = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(raw):
        if not line.strip():
            continue
        try:
            lines.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(raw) - 1:
                raise
            # A live writer may have one incomplete final record.
    header, records = lines[0], lines[1:]
    stages = header["stages"]
    frames = [r for r in records if "ms" in r]
    startup = [r for r in records if "startup" in r]
    by_scenario = defaultdict(list)
    for r in frames:
        if r["time_ms"] >= 2000:
            by_scenario[r.get("scenario", "unknown")].append(r)
    groups = {}
    for name, samples in by_scenario.items():
        groups[name] = {
            "frames": len(samples),
            "stages_ms": {s: stats([r["ms"][i] for r in samples]) for i, s in enumerate(stages)},
            "calls": {s: stats([r.get("calls", [0] * len(stages))[i] for r in samples]) for i, s in enumerate(stages)},
            "active_gap_ms": stats([r["gap_ms"] for previous, r in zip(samples, samples[1:])
                                    if previous.get("active") and r.get("active") and r["gap_ms"] > 0]),
            "continuous_gap_ms": stats([r["gap_ms"] for r in samples if header.get("continuous") and r["time_ms"] < 19000 and r.get("pass", 0) == 0]),
            "previous_frame_cpu_ms": stats([r["previous_cpu_ms"] for r in samples if r.get("previous_cpu_ms") is not None]),
        }
    total = stages.index("total")
    grouped = defaultdict(list)
    for index, r in enumerate(frames):
        grouped[r.get("frame_nr", index)].append(r)
    groups_in_order = list(grouped.values())
    overhead = [following[0]["previous_cpu_ms"] - sum(r["ms"][total] for r in previous)
                for previous, following in zip(groups_in_order, groups_in_order[1:])
                if following[0].get("previous_cpu_ms") is not None]
    return {
        "path": str(path), "passes": len(frames), "frames": len(grouped), "startup": startup,
        "first_ui_ms": frames[0]["time_ms"] if frames else None,
        "first_ui_work_ms": frames[0]["ms"][total] if frames else None,
        "outside_app_ui_cpu_ms": stats(overhead), "scenarios": groups,
        "slowest_passes": sorted(frames, key=lambda r: r["ms"][total], reverse=True)[:8],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps([analyse(path) for path in args.traces], indent=2)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
