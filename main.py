"""
main.py — Entry point for the Multi-Agent Software Engineering System.

Usage:
    python main.py --issue https://github.com/owner/repo/issues/42

This runs the full pipeline:
    GitHub Issue → Orchestrator → Research → Code → Test → (retry loop) → Open PR

Prerequisites:
    1. Set GOOGLE_API_KEY and GITHUB_TOKEN in .env file
    2. Docker Desktop must be running
    3. pip install -r requirements.txt
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from config import LOG_FORMAT, LOG_LEVEL, run_all_health_checks
from src.graph.pipeline import run_pipeline

logger = logging.getLogger(__name__)


def main():
    """
    Parse CLI arguments and run the full pipeline.

    Args (CLI):
        --issue: GitHub issue URL (required)
        --verbose: Enable debug logging (optional)
    """
    parser = argparse.ArgumentParser(
        description="Multi-Agent Software Engineering System",
        epilog=(
            "Example: python main.py "
            "--issue https://github.com/owner/repo/issues/42"
        ),
    )
    parser.add_argument(
        "--issue",
        type=str,
        required=True,
        help="Full GitHub issue URL to resolve",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode (simulates agents without real API calls)",
    )
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, force=True)
    else:
        logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, force=True)

    # Banner
    print("=" * 70)
    print("  Multi-Agent Software Engineering System v1.0.0")
    print("  Powered by LangGraph + Gemini")
    print("=" * 70)
    print(f"\n  Issue: {args.issue}\n")

    # Health checks
    if args.mock:
        print("⚠️  MOCK MODE: Skipping real health checks.\n")
    else:
        print("Running health checks...")
        health = run_all_health_checks()
        if not health.get("google_api_key"):
            print("\n[ERROR] GOOGLE_API_KEY not set. See .env.example for instructions.")
            sys.exit(1)
        if not health.get("github_token"):
            print("\n[ERROR] GITHUB_TOKEN not set. See .env.example for instructions.")
            sys.exit(1)
        if not health.get("docker"):
            print("\n[WARNING] Docker is not available. Sandbox testing will be skipped.")
            print("   (To enable safe testing, start Docker Desktop and retry)\n")
        
        print("[OK] Health checks passed.\n")
    print("Starting pipeline...\n")

    # Run the pipeline
    try:
        result = run_pipeline(args.issue, use_mock=args.mock)
    except Exception as e:
        logger.error("Pipeline crashed: %s", e, exc_info=True)
        print(f"\n[ERROR] Pipeline crashed: {e}")
        sys.exit(1)

    # Display results
    status = result.get("pipeline_status", "unknown")
    print("\n" + "=" * 70)

    if status == "done":
        pr_url = result.get("pr_url", "N/A")
        print("  [OK] SUCCESS — Pull request opened!")
        print(f"  [Link] PR URL: {pr_url}")
        print(f"  [Fix] explanation: {result.get('fix_explanation', 'N/A')[:200]}")
        print(f"  [Test] results: {result.get('execution_result', {}).get('tests_passed', 0)} passed")
    elif status == "failed":
        print("  [FAIL] FAILED — Could not resolve the issue automatically.")
        print(f"  Retries: {result.get('retry_count', 0)}")
        print(f"  Reason: {result.get('failure_reason', 'unknown')[:300]}")
    else:
        print(f"  [WARN] Unexpected status: {status}")

    print("=" * 70)

    # Save full state to file for debugging
    output_file = Path("pipeline_output.json")
    # Serialize the result, handling non-serializable types
    serializable = {}
    for k, v in result.items():
        try:
            json.dumps(v)
            serializable[k] = v
        except (TypeError, ValueError):
            serializable[k] = str(v)

    output_file.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"\nFull pipeline state saved to: {output_file}")

    return 0 if status == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
