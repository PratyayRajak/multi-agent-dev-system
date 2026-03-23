"""
src/agents/researcher.py — Research Agent.

Reads:  state.issue_title, state.issue_body, state.repo_owner, state.repo_name
Writes: state.relevant_files, state.pipeline_status = 'coding'

Role:
    - Searches the GitHub repo for files related to the issue.
    - Reads up to MAX_FILES_TO_READ files.
    - Identifies suspected functions/classes responsible for the bug.
    - Uses Gemini LLM to reason about which files are relevant.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import AGENT_TEMPERATURES, GEMINI_MODEL, MAX_FILES_TO_READ
from src.tools.github_tools import get_file_content, list_directory, search_code

logger = logging.getLogger(__name__)

# Load system prompt
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "researcher.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


def _build_context(state: Dict[str, Any]) -> str:
    """Build the user message with issue context for the LLM."""
    repo_full = f"{state['repo_owner']}/{state['repo_name']}"
    return (
        f"## GitHub Issue\n"
        f"**Repository:** {repo_full}\n"
        f"**Title:** {state.get('issue_title', 'N/A')}\n"
        f"**Body:**\n{state.get('issue_body', 'No description provided.')}\n\n"
        f"## Task\n"
        f"Search the repository and identify all files relevant to this bug. "
        f"Return your findings as a JSON list."
    )


def research_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Research Agent node for the LangGraph pipeline.

    Reads from state:
        - issue_title, issue_body: What the bug is about.
        - repo_owner, repo_name: Which repository to search.

    Writes to state:
        - relevant_files: List of dicts with {path, content, reason, suspected_function}.
        - pipeline_status: Set to 'coding'.

    Args:
        state: Current pipeline state dict.

    Returns:
        Updated state dict with relevant_files populated.
    """
    repo_owner = state["repo_owner"]
    repo_name = state["repo_name"]
    repo_full = f"{repo_owner}/{repo_name}"
    issue_title = state.get("issue_title", "")
    issue_body = state.get("issue_body", "")

    logger.info("=" * 60)
    logger.info("RESEARCHER: Analyzing issue — '%s'", issue_title)
    logger.info("=" * 60)

    # Step 1: Get repo structure
    try:
        root_files = list_directory(repo_full, "")
        logger.info("Root directory: %s", root_files)
    except Exception as e:
        logger.warning("Could not list root directory: %s", e)
        root_files = []

    # Step 2: Search for keywords from the issue
    keywords = _extract_keywords(issue_title, issue_body)
    search_results = []
    for keyword in keywords[:5]:  # Search top 5 keywords
        try:
            results = search_code(repo_full, keyword)
            search_results.extend(results)
            logger.info("Search '%s': found %d results", keyword, len(results))
        except Exception as e:
            logger.warning("Search for '%s' failed: %s", keyword, e)

    # Deduplicate by path
    seen_paths = set()
    unique_results = []
    for r in search_results:
        if r["path"] not in seen_paths:
            seen_paths.add(r["path"])
            unique_results.append(r)

    # Step 3: Read file content for top results
    relevant_files = []
    for result in unique_results[:MAX_FILES_TO_READ]:
        try:
            content = get_file_content(repo_full, result["path"])
            relevant_files.append({
                "path": result["path"],
                "content": content,
                "reason": f"Found via code search — matched keyword in issue",
                "suspected_function": None,
            })
            logger.info("Read file: %s (%d chars)", result["path"], len(content))
        except Exception as e:
            logger.warning("Could not read %s: %s", result["path"], e)

    # Step 4: Use LLM to refine analysis — identify suspected functions
    if relevant_files:
        relevant_files = _refine_with_llm(state, relevant_files)

    logger.info("RESEARCHER: Found %d relevant files", len(relevant_files))

    return {
        "relevant_files": relevant_files,
        "pipeline_status": "coding",
    }


def _extract_keywords(title: str, body: str) -> List[str]:
    """
    Extract search keywords from issue title and body.
    Simple heuristic — extracts words that look like function/class/file names.
    """
    text = f"{title} {body}"
    # Look for function-like words, file paths, and significant identifiers
    words = text.split()
    keywords = []
    for word in words:
        clean = word.strip(".,;:!?()[]{}\"'`")
        # Keep words that look like identifiers (contain underscore, or are CamelCase)
        if (
            "_" in clean
            or (clean and clean[0].isupper() and any(c.islower() for c in clean))
            or clean.endswith(".py")
            or clean.endswith(".js")
            or clean.endswith(".ts")
        ):
            keywords.append(clean)
    # Also add significant short phrases from the title
    title_words = [w.strip(".,;:!?()[]{}\"'`") for w in title.split()]
    keywords.extend([w for w in title_words if len(w) > 3])
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for k in keywords:
        if k.lower() not in seen:
            seen.add(k.lower())
            unique.append(k)
    return unique


def _refine_with_llm(
    state: Dict[str, Any], files: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Use Gemini to analyze retrieved files and identify suspected functions.
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=AGENT_TEMPERATURES["researcher"],
        )

        file_summaries = []
        for f in files:
            content_preview = f["content"][:2000]  # Limit context size
            file_summaries.append(
                f"### {f['path']}\n```\n{content_preview}\n```"
            )

        user_msg = (
            f"## Issue\n"
            f"**Title:** {state.get('issue_title', '')}\n"
            f"**Body:** {state.get('issue_body', '')}\n\n"
            f"## Files Found\n"
            f"{''.join(file_summaries)}\n\n"
            f"## Task\n"
            f"For each file, explain WHY it is relevant to the bug and identify "
            f"the specific function or class most likely responsible.\n"
            f"Return a JSON array where each element has: "
            f"path, reason, suspected_function.\n"
            f"Return ONLY the JSON array, no other text."
        )

        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])

        # Parse LLM response
        response_text = response.content.strip()
        # Strip markdown code fences if present
        if response_text.startswith("```"):
            response_text = response_text.split("\n", 1)[1]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

        analysis = json.loads(response_text)

        # Merge LLM analysis back into files
        analysis_map = {item["path"]: item for item in analysis}
        for f in files:
            if f["path"] in analysis_map:
                info = analysis_map[f["path"]]
                f["reason"] = info.get("reason", f["reason"])
                f["suspected_function"] = info.get("suspected_function")

        logger.info("LLM analysis complete — refined %d files", len(files))

    except Exception as e:
        logger.warning("LLM refinement failed (continuing without): %s", e)

    return files
