"""
runs_log.py — Logs every pipeline run to runs_log.csv.
Import and call log_run() at the end of each pipeline execution.
This file generates the success rate metric for the resume.
"""

import csv
import os
from datetime import datetime

LOG_FILE = "runs_log.csv"

HEADERS = [
    "timestamp",
    "issue_url",
    "repo",
    "issue_number",
    "status",           # success | failed
    "retry_count",
    "time_taken_seconds",
    "pr_url",
    "failure_reason",
]


def _ensure_headers():
    """Create CSV with headers if it doesn't exist yet."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=HEADERS)
            writer.writeheader()


def log_run(
    issue_url: str,
    status: str,
    time_taken_seconds: float,
    retry_count: int = 0,
    pr_url: str = "",
    failure_reason: str = "",
):
    """
    Log one pipeline run to runs_log.csv.

    Call this at the end of every pipeline execution, success or failure.

    Args:
        issue_url:           Full GitHub issue URL
        status:              "success" or "failed"
        time_taken_seconds:  How long the pipeline took
        retry_count:         How many times Coder retried
        pr_url:              URL of opened PR (empty if failed)
        failure_reason:      Why it failed (empty if success)
    """
    _ensure_headers()

    # Parse repo and issue number from URL
    # e.g. https://github.com/owner/repo/issues/42
    parts = issue_url.rstrip("/").split("/")
    try:
        repo = f"{parts[-4]}/{parts[-3]}"
        issue_number = parts[-1]
    except IndexError:
        repo = "unknown"
        issue_number = "unknown"

    row = {
        "timestamp":           datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "issue_url":           issue_url,
        "repo":                repo,
        "issue_number":        issue_number,
        "status":              status,
        "retry_count":         retry_count,
        "time_taken_seconds":  round(time_taken_seconds, 1),
        "pr_url":              pr_url,
        "failure_reason":      failure_reason,
    }

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writerow(row)

    print(f"[runs_log] Logged: {status.upper()} | {repo}#{issue_number} | {round(time_taken_seconds)}s")


def print_summary():
    """Print success rate and avg time from runs_log.csv."""
    if not os.path.exists(LOG_FILE):
        print("No runs logged yet.")
        return

    with open(LOG_FILE, "r") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No runs logged yet.")
        return

    total = len(rows)
    successes = [r for r in rows if r["status"] == "success"]
    failures  = [r for r in rows if r["status"] == "failed"]
    success_rate = round(len(successes) / total * 100, 1)

    times = [float(r["time_taken_seconds"]) for r in successes if r["time_taken_seconds"]]
    avg_time = round(sum(times) / len(times), 1) if times else 0

    print("\n========== PIPELINE SUMMARY ==========")
    print(f"Total runs    : {total}")
    print(f"Successes     : {len(successes)}")
    print(f"Failures      : {len(failures)}")
    print(f"Success rate  : {success_rate}%")
    print(f"Avg time (success): {avg_time}s ({round(avg_time/60, 1)} min)")
    if failures:
        print("\nFailure reasons:")
        for r in failures:
            print(f"  - {r['repo']}#{r['issue_number']}: {r['failure_reason']}")
    print("======================================\n")


if __name__ == "__main__":
    print_summary()
