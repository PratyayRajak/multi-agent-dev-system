"""
src/agents/orchestrator.py — Orchestrator Agent (Manager node).

Reads:  state.issue_url
Writes: state.issue_title, state.issue_body, state.issue_number,
        state.issue_comments, state.repo_owner, state.repo_name,
        state.pipeline_status

Role:
    - Parses the GitHub issue URL to extract owner, repo, issue number.
    - Fetches the full issue via GitHub API.
    - Sets pipeline_status to 'researching' so the graph routes to Research Agent.
    - Does NOT do any real coding/testing work — just orchestration.
"""

import logging
import re
from typing import Any, Dict

from src.tools.github_tools import get_issue

logger = logging.getLogger(__name__)


def parse_issue_url(url: str) -> Dict[str, Any]:
    """
    Parse a GitHub issue URL into its components.

    Args:
        url: Full GitHub issue URL, e.g. 'https://github.com/owner/repo/issues/42'

    Returns:
        dict with keys: owner, repo, issue_number

    Raises:
        ValueError: If the URL format is invalid.
    """
    pattern = r"https?://github\.com/([^/]+)/([^/]+)/issues/(\d+)"
    match = re.match(pattern, url.strip())
    if not match:
        raise ValueError(
            f"Invalid GitHub issue URL: '{url}'. "
            "Expected format: https://github.com/owner/repo/issues/123"
        )
    return {
        "owner": match.group(1),
        "repo": match.group(2),
        "issue_number": int(match.group(3)),
    }


def orchestrator_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrator node for the LangGraph pipeline.

    Reads from state:
        - issue_url: The GitHub issue URL provided by the caller.

    Writes to state:
        - repo_owner, repo_name, issue_number: Parsed from URL.
        - issue_title, issue_body, issue_comments: Fetched from GitHub API.
        - pipeline_status: Set to 'researching'.

    Args:
        state: Current pipeline state dict.

    Returns:
        Updated state dict.
    """
    issue_url = state["issue_url"]
    logger.info("=" * 60)
    logger.info("ORCHESTRATOR: Starting pipeline for %s", issue_url)
    logger.info("=" * 60)

    # Step 1: Parse the issue URL
    parsed = parse_issue_url(issue_url)
    owner = parsed["owner"]
    repo = parsed["repo"]
    issue_number = parsed["issue_number"]

    logger.info("Parsed: owner=%s, repo=%s, issue=#%d", owner, repo, issue_number)

    # Step 2: Fetch issue details from GitHub
    issue_data = get_issue(owner, repo, issue_number)

    logger.info("Issue title: %s", issue_data["title"])
    logger.info("Issue has %d comments", len(issue_data.get("comments", [])))

    # Step 3: Update state and set status
    return {
        "repo_owner": owner,
        "repo_name": repo,
        "issue_number": issue_number,
        "issue_title": issue_data["title"],
        "issue_body": issue_data["body"],
        "issue_comments": issue_data.get("comments", []),
        "pipeline_status": "researching",
    }
