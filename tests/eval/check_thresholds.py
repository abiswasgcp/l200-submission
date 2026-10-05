"""CI eval gate: fail the build if the latest eval results miss the quality bar.

`agents-cli eval run` exits 0 whatever the scores are, so CI runs this after it.

Usage: uv run python tests/eval/check_thresholds.py [results.json]
"""

import glob
import json
import sys

THRESHOLDS = {
    "custom_response_quality": 4.0,  # mean LLM-judge score (1-5)
    "allergen_safety": 1.0,  # every case must be allergen-safe
}


def main() -> int:
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else max(glob.glob("artifacts/grade_results/results_*.json"), default=None)
    )
    if not path:
        print("No eval results found.")
        return 1
    summary = {m["metric_name"]: m for m in json.load(open(path))["summary_metrics"]}
    failed = False
    for metric, bar in THRESHOLDS.items():
        m = summary.get(metric)
        score = m and m.get("mean_score")
        errors = m.get("num_cases_error", 0) if m else 0
        ok = score is not None and score >= bar and errors == 0
        failed |= not ok
        print(
            f"{'PASS' if ok else 'FAIL'} {metric}: mean={score} (bar {bar}, errors {errors})"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
