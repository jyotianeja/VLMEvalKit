#!/usr/bin/env python3
"""Collect and display final scores from all benchmark results in outputs/."""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_OUTPUTS_DIR = Path(__file__).parent / "outputs/"


def extract_benchmark_name(filename: str, model_prefix: str) -> str:
    """Extract benchmark name from filename by stripping model prefix and suffix."""
    name = filename.removeprefix(model_prefix + "_")
    # Remove known suffixes
    # Remove score file suffixes, including judge-prefixed ones like _gpt-4o_score.csv
    name = re.sub(r"(_gpt-[a-zA-Z0-9\-]+)?_score\.(json|csv)$", "", name)
    name = re.sub(r"_acc\.csv$", "", name)
    return name


def parse_acc_csv(filepath: Path) -> dict[str, float]:
    """Parse an _acc.csv file and return {metric: value} dict."""
    scores = {}
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get("split", "")
            overall = row.get("Overall")
            if overall is not None:
                val = float(overall)
                # If value is <= 1, it's a ratio; convert to percentage
                if val <= 1.0:
                    val *= 100
                label = "Overall" if split in ("none", "") else f"Overall ({split})"
                # Skip non-validation splits for MMMU
                if split == "dev":
                    continue
                scores[label] = val
    return scores


def parse_score_json(filepath: Path) -> dict[str, float]:
    """Parse a _score.json file and return the key scores."""
    with open(filepath) as f:
        data = json.load(f)
    scores = {}
    # Pick the single best summary key in priority order
    for key in ("Final Score Norm", "Overall_Accuracy"):
        if key in data:
            scores[key] = float(data[key])
            return scores
    # Fallback: grab all numeric values (skip counts)
    for k, v in data.items():
        if isinstance(v, (int, float)) and "cnt" not in k.lower():
            scores[k] = float(v)
    return scores


def parse_score_csv(filepath: Path) -> dict[str, float]:
    """Parse a _score.csv file and return Overall scores.

    Handles multiple formats:
    - HallusionBench: split=Overall, columns aAcc/fAcc/qAcc
    - MathVista/MathVision: Task&Skill/Subject=Overall, column acc
    - MathVerse: split column with no Overall row; average the Overall column across splits
    """
    scores = {}
    # Columns that identify the row label (not a score)
    label_cols = {"split", "Task&Skill", "Subject"}
    # Score columns to look for in an Overall row
    overall_score_cols = ("aAcc", "acc")

    with open(filepath) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return scores

    # Find which column is the label column
    header_set = set(rows[0].keys())
    label_col = None
    for col in label_cols:
        if col in header_set:
            label_col = col
            break

    # Look for an explicit "Overall" row
    for row in rows:
        if label_col and row.get(label_col) == "Overall":
            for col in overall_score_cols:
                if col in row and row[col]:
                    scores[col] = float(row[col])
            # Also check for an "Overall" value column (e.g. MathVerse)
            if "Overall" in row and row["Overall"]:
                scores["Overall"] = float(row["Overall"])
            return scores

    # No "Overall" row found — if there's an "Overall" column, average it across all rows
    if "Overall" in header_set:
        vals = []
        for row in rows:
            if row.get("Overall"):
                vals.append(float(row["Overall"]))
        if vals:
            scores["Overall (avg splits)"] = sum(vals) / len(vals)

    return scores


def collect_model_scores(model_dir: Path) -> list[tuple[str, str, float]]:
    """Collect all (benchmark, metric, score) tuples for a model directory."""
    model_name = model_dir.name
    results = []

    for f in sorted(model_dir.iterdir()):
        if f.name.endswith("_acc.csv"):
            bench = extract_benchmark_name(f.name, model_name)
            for metric, val in parse_acc_csv(f).items():
                results.append((bench, metric, val))
        elif f.name.endswith("_score.json"):
            bench = extract_benchmark_name(f.name, model_name)
            for metric, val in parse_score_json(f).items():
                results.append((bench, metric, val))
        elif f.name.endswith("_score.csv"):
            bench = extract_benchmark_name(f.name, model_name)
            for metric, val in parse_score_csv(f).items():
                results.append((bench, metric, val))

    return results


def compute_family_averages(rows: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
    """Compute averages for benchmark families (e.g. ScreenSpot_v2_Desktop/Mobile/Web).

    A family is a group of benchmarks sharing a common prefix (everything up to
    the last '_'-separated segment) where at least 2 benchmarks share that prefix
    and the same metric name.
    """
    # Group by (family_prefix, metric) -> list of scores
    family_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for bench, metric, score in rows:
        # Strip _tmp suffix before detecting families
        name = bench.removesuffix("_tmp")
        last_underscore = name.rfind("_")
        if last_underscore > 0:
            prefix = name[:last_underscore]
            family_scores[(prefix, metric)].append(score)

    averages = []
    seen = set()
    for (prefix, metric), scores in sorted(family_scores.items()):
        if len(scores) >= 2 and prefix not in seen:
            seen.add(prefix)
            averages.append((f"{prefix} (avg)", metric, sum(scores) / len(scores)))
    return averages


def print_combined_table(
    model_scores: dict[str, list[tuple[str, str, float]]],
):
    """Print a combined table with benchmarks as rows and models as columns."""
    # Collect all (bench, metric) pairs and per-model scores including family averages
    # key: (bench, metric) -> {model: score}
    score_map: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    model_names = sorted(model_scores.keys())

    for model, rows in model_scores.items():
        averages = compute_family_averages(rows)
        for bench, metric, score in rows + averages:
            score_map[(bench, metric)][model] = score

    if not score_map:
        print("No results found.")
        return

    # Sort benchmark rows: regular first, then averages
    all_keys = sorted(score_map.keys(), key=lambda k: (k[0].endswith("(avg)"), k[0].lower(), k[1].lower()))

    # Column widths
    bench_w = max(max(len(k[0]) for k in all_keys), len("Benchmark"))
    metric_w = max(max(len(k[1]) for k in all_keys), len("Metric"))
    max_col_w = 10
    col_w = max(min(max(len(m) for m in model_names), max_col_w), 8)

    # Header: wrap model names into multiple lines of col_w chars each
    def wrap_name(name: str, width: int) -> list[str]:
        chunks = []
        while name:
            chunks.append(name[:width])
            name = name[width:]
        return chunks

    wrapped = [wrap_name(m, col_w) for m in model_names]
    max_lines = max(len(w) for w in wrapped)
    # Pad shorter wraps with empty strings at the top
    for w in wrapped:
        while len(w) < max_lines:
            w.insert(0, "")

    prefix_w = bench_w + metric_w + 6  # "  " + bench + "  " + metric + "  "
    for i in range(max_lines):
        line = " " * prefix_w
        for w in wrapped:
            line += f"{w[i]:>{col_w}}  "
        print(line.rstrip())

    sep = f"  {'-' * bench_w}  {'-' * metric_w}"
    for _ in model_names:
        sep += f"  {'-' * col_w}"
    print(sep)

    # Rows
    prev_is_avg = False
    for bench, metric in all_keys:
        is_avg = bench.endswith("(avg)")
        if is_avg and not prev_is_avg:
            print(sep)
        prev_is_avg = is_avg

        line = f"  {bench:<{bench_w}}  {metric:<{metric_w}}"
        for m in model_names:
            if m in score_map[(bench, metric)]:
                line += f"  {score_map[(bench, metric)][m]:>{col_w}.1f}"
            else:
                line += f"  {'—':>{col_w}}"
        print(line)
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collect and display benchmark scores.")
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR,
                        help="Directory containing model output subdirectories.")
    args = parser.parse_args()
    outputs_dir = args.outputs_dir

    if not outputs_dir.exists():
        print(f"No outputs directory found at {outputs_dir}")
        sys.exit(1)

    model_dirs = sorted(
        [d for d in outputs_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    if not model_dirs:
        print("No model output directories found.")
        sys.exit(1)

    model_scores = {}
    for model_dir in model_dirs:
        model_scores[model_dir.name] = collect_model_scores(model_dir)

    print_combined_table(model_scores)


if __name__ == "__main__":
    main()
