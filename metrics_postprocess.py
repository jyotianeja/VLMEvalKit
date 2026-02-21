import os, glob
from collections import deque
import re
import string
import pprint
import sys

# Define keywords for metric extraction
metrics_keywords = {"overall_accuracy", "overall", "accuracy", "score (strict)", "win rate", "final score norm"}

# Define file paths
log_dir = sys.argv[1]
files = glob.glob(os.path.join(log_dir, "*"))

def extract_matching_lines(file_path, keywords):
    """
    Returns a list of tuples:
    (line, matched_word)

    - Case-insensitive
    - Whole-word match
    - Reads last 50 lines only
    """
    pattern = re.compile(
        r'\b(' + '|'.join(map(re.escape, metrics_keywords)) + r')(?!\w)',
        re.IGNORECASE
    )

    results = []

    with open(file_path, 'r', encoding='utf-8') as f:
        last_50_lines = deque(f, maxlen=50)

    for line in last_50_lines:
        match = pattern.search(line)
        if match:
            results.append((line.strip(), match.group(0)))

    return results

def process_metric_output(line, keyword, benchmark_name):
    line = line.lower()
    keyword = keyword.lower()
    assert keyword in line, f"Keyword '{keyword}' not found in line: {line}"
    
    metric_value = line.split(keyword)[1:]
    assert len(metric_value) == 1, f"Expected exactly one metric value after keyword '{keyword}' in line: {line}"
    metric_value = metric_value[0].strip().strip(string.punctuation)
    
    try:
        metric_value = float(metric_value)
    except ValueError:
        
        # Try benchmark-specific parsing if the metric value is not a simple float
        if benchmark_name == "MMMU_DEV_VAL":
            """
            split                                validation           dev
            Overall                              0.5788888888888889   0.5533333333333333
            """
            metric_value = float(metric_value.split()[1])
        elif benchmark_name in ("HallusionBench", "MathVista_MINI", "MathVision_MINI"):
            """
            [HallusionBench]: Overall      65.6151  38.7283  40.2198
            [MathVista_MINI, MathVision_MINI]: Overall      1000  608  733  60.8     73.3
            """
            metric_value = float(metric_value.split()[-1])
        elif benchmark_name == "MathVerse_MINI":
            """
            split           Vision Dominant     Text Dominant      Text Lite           Vision Intensive    Vision Only
            Overall         47.08121827411168   66.6243654822335   56.09137055837563   50.380710659898476  48.223350253807105
            """
            # Average of all splits
            metric_value = sum(float(x) for x in metric_value.split()) / 5
            
        else:
            raise ValueError(f"Could not convert metric value to float: '{metric_value}' in line: {line}")

    return metric_value

final_metrics = {}

for file in files:
    # Read the first line and extract the benchmark name (format is === START [benchmark_name] ===)
    with open(file, 'r') as f:
        first_line = f.readline().strip()
        match = re.search(r'=== START (.*?) on port \d+ ===', first_line)
        if match:
            benchmark_name = match.group(1)
        else:
            raise ValueError(f"First line of file {file} does not match expected format: {first_line}")
    
    results = extract_matching_lines(file, metrics_keywords)
    
    if len(results) != 1:
        # Try benchmark-specific parsing if we don't find exactly one metric line using the general keywords
        if benchmark_name == "WeMath":
            results = results[:1]
        else:
            raise ValueError(f"Expected exactly one matching metric in {file}, got {len(results)}")
        
    line_output, matched_metric = results[0]
    final_metrics[benchmark_name] = process_metric_output(line_output, matched_metric, benchmark_name)

# Benchmark list and formatting
benchmark_names = [
    "AI2D_TEST", "BLINK", "ChartQA_TEST", "DocVQA_VAL", "HallusionBench",
    "MathVerse_MINI", "MathVision_MINI", "MathVista_MINI", "MMMU_DEV_VAL",
    "MMStar", "OCRBench", "ScreenSpot_v2_Desktop", "ScreenSpot_v2_Mobile",
    "ScreenSpot_v2_Web", "WeMath", "WildVision", "ZEROBench_sub"
]

formatted = [
    round(final_metrics[name] * 100 if final_metrics[name] < 1 else final_metrics[name], 1)
    # final_metrics[name] * 100 if final_metrics[name] < 1 else final_metrics[name]
    for name in benchmark_names
]

pprint.pprint(formatted)
