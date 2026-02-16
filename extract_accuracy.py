#!/usr/bin/env python3
"""
Extract overall accuracy from VLMEvalKit log files.
Handles multiple output formats: JSON results, tabular results, and edge cases.
"""

import re
import json
import glob
import os
import sys


def extract_benchmark_name(filepath):
    """Extract the benchmark name from the filename."""
    basename = os.path.basename(filepath)
    # Pattern: BenchmarkName_model...
    match = re.match(r'^(.+?)_model', basename)
    return match.group(1) if match else basename


def try_json_overall(content):
    """Try to extract Overall_Accuracy from JSON block."""
    # Find JSON blocks after "Evaluation Results"
    pattern = r'Evaluation Results:.*?\n\{(.*?)\}'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return None

    try:
        json_str = '{' + match.group(1) + '}'
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    # Look for top-level "Overall_Accuracy" (not prefixed with sub-category)
    if "Overall_Accuracy" in data:
        val = data["Overall_Accuracy"]
        # Convert 0-1 range to percentage
        return val * 100 if val <= 1.0 else val

    # OCRBench uses "Final Score Norm"
    if "Final Score Norm" in data:
        return data["Final Score Norm"]

    # If only sub-category Overall_Accuracy keys exist (e.g. ScreenSpot_Pro),
    # average them
    sub_overalls = {k: v for k, v in data.items()
                    if k.endswith(":Overall_Accuracy")}
    if sub_overalls:
        vals = [v * 100 if v <= 1.0 else v for v in sub_overalls.values()]
        return sum(vals) / len(vals)

    return None


def try_table_overall(content):
    """Try to extract Overall accuracy from tabular output."""
    # Look for lines after "Evaluation Results:" that contain "Overall"
    results_match = re.search(r'Evaluation Results:.*?\n(.*?)(?:\n=== END|\Z)',
                              content, re.DOTALL)
    if not results_match:
        return None

    results_block = results_match.group(1)

    for line in results_block.split('\n'):
        if 'Overall' not in line:
            continue

        # Split by whitespace and find numeric values
        parts = line.split()
        # Find the index of "Overall"
        try:
            idx = parts.index('Overall')
        except ValueError:
            continue

        # Grab numbers after "Overall"
        numbers = []
        for p in parts[idx + 1:]:
            try:
                numbers.append(float(p))
            except ValueError:
                continue

        if not numbers:
            continue

        # If the first number is > 100, it's likely a count (e.g. total samples).
        # In that case, the accuracy is the LAST number on the line.
        if numbers[0] > 100:
            val = numbers[-1]
        else:
            val = numbers[0]

        # If it looks like a 0-1 fraction, convert to percentage
        if val <= 1.0:
            val *= 100
        return val

    return None


def check_failed(content):
    """Check if the run failed (download error, parse error, etc.)."""
    if 'combination failed' in content:
        # Extract reason
        match = re.search(r'combination failed: (.+?)(?:,|\n)', content)
        reason = match.group(1).strip() if match else "Unknown error"
        return reason
    return None


def extract_accuracy(filepath):
    """Extract overall accuracy from a log file. Returns (value, note)."""
    with open(filepath, 'r', errors='replace') as f:
        content = f.read()

    # Check for failures first
    failure = check_failed(content)
    if failure:
        return None, f"FAILED: {failure}"

    # Check if "Evaluation Results" exists at all
    if 'Evaluation Results' not in content:
        return None, "No evaluation results found"

    # Try JSON format first
    val = try_json_overall(content)
    if val is not None:
        return val, ""

    # Try table format
    val = try_table_overall(content)
    if val is not None:
        return val, ""

    return None, "Could not parse results"


def main():
    # Find all log files
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads"
    log_files = sorted(glob.glob(os.path.join(log_dir, "*.log")))

    if not log_files:
        print(f"No .log files found in {log_dir}")
        return

    print(f"{'Benchmark':<35} {'Overall Accuracy':>18}  Notes")
    print("-" * 80)

    for filepath in log_files:
        name = extract_benchmark_name(filepath)
        accuracy, note = extract_accuracy(filepath)

        if accuracy is not None:
            print(f"{name:<35} {accuracy:>17.2f}%  {note}")
        else:
            print(f"{name:<35} {'N/A':>18}  {note}")


if __name__ == "__main__":
    main()
