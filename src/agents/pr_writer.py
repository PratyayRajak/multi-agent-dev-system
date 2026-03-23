"""
src/agents/pr_writer.py — PR Writer Agent.

Reads:  All state fields — issue context, fix, tests, execution results.
Writes: state.pr_url, state.pipeline_status = 'done'

Role:
    - Creates a new branch: fix/issue-{issue_number}
    - Commits the fixed files to the branch
    - Writes a professional PR title and body
    - Opens the pull request via GitHub API
    - Returns the PR URL
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import (
    AGENT_TEMPERATURES,
    GEMINI_MODEL,
    GITHUB_BASE_BRANCH,
    GITHUB_FIX_BRANCH_PREFIX,
)
from src.tools.github_tools import (
    commit_file,
    create_branch,
    get_file_content,
    open_pull_request,
)

logger = logging.getLogger(__name__)

# Load system prompt
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "pr_writer.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


def pr_writer_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    PR Writer Agent node for the LangGraph pipeline.

    Reads from state:
        - repo_owner, repo_name: Target repository.
        - issue_number, issue_title, issue_body: Issue context.
        - relevant_files: Files that were analysed.
        - proposed_fix: The unified diff fix.
        - fix_explanation: Why the fix works.
        - test_code: Tests that were written and passed.
        - execution_result: Test results.

    Writes to state:
        - pr_url: URL of the opened pull request.
        - pipeline_status: Set to 'done'.

    Args:
        state: Current pipeline state dict.

    Returns:
        Updated state dict with pr_url and pipeline_status='done'.
    """
    repo_owner = state["repo_owner"]
    repo_name = state["repo_name"]
    repo_full = f"{repo_owner}/{repo_name}"
    issue_number = state["issue_number"]
    branch_name = f"{GITHUB_FIX_BRANCH_PREFIX}{issue_number}"

    logger.info("=" * 60)
    logger.info("PR WRITER: Creating pull request for issue #%d", issue_number)
    logger.info("=" * 60)

    # Step 1: Create the fix branch
    logger.info("PR WRITER: Creating branch '%s'", branch_name)
    create_branch(repo_full, branch_name, from_branch=GITHUB_BASE_BRANCH)

    # Step 2: Apply the fix — parse the diff and commit changed files
    _apply_fix_to_branch(state, repo_full, branch_name)

    # Step 3: Commit test file
    if state.get("test_code"):
        test_path = _determine_test_path(state)
        logger.info("PR WRITER: Committing test file: %s", test_path)
        commit_file(
            repo_full_name=repo_full,
            branch=branch_name,
            file_path=test_path,
            new_content=state["test_code"],
            commit_message=f"test: add tests for issue #{issue_number}",
        )

    # Step 4: Generate PR title and body using LLM
    pr_title, pr_body = _generate_pr_content(state)

    # Step 5: Open the pull request
    logger.info("PR WRITER: Opening pull request...")
    pr_url = open_pull_request(
        repo_full_name=repo_full,
        title=pr_title,
        body=pr_body,
        head_branch=branch_name,
        base_branch=GITHUB_BASE_BRANCH,
    )

    logger.info("PR WRITER: Pull request opened: %s", pr_url)
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE — PR URL: %s", pr_url)
    logger.info("=" * 60)

    return {
        "pr_url": pr_url,
        "pipeline_status": "done",
    }


def _apply_fix_to_branch(
    state: Dict[str, Any], repo_full: str, branch_name: str
) -> None:
    """
    Parse the proposed fix (unified diff) and commit the changed files.
    """
    proposed_fix = state.get("proposed_fix", "")
    issue_number = state.get("issue_number", 0)

    # Parse diff to find which files to update
    file_changes = _parse_diff(proposed_fix, state)

    for file_path, new_content in file_changes.items():
        logger.info("PR WRITER: Committing fix to %s", file_path)
        commit_file(
            repo_full_name=repo_full,
            branch=branch_name,
            file_path=file_path,
            new_content=new_content,
            commit_message=f"fix: resolve issue #{issue_number} in {file_path}",
        )


def _parse_diff(diff_text: str, state: Dict[str, Any]) -> Dict[str, str]:
    """
    Parse a unified diff and apply changes to the original file content.

    Returns:
        dict of {file_path: updated_content}
    """
    file_changes = {}

    # Build a map of original file contents
    original_files = {}
    for f in state.get("relevant_files", []):
        fd = f if isinstance(f, dict) else f.dict() if hasattr(f, 'dict') else {}
        path = fd.get("path", "")
        content = fd.get("content", "")
        if path:
            original_files[path] = content

    # Try to parse unified diff format
    # Look for --- a/path and +++ b/path patterns
    current_file = None
    current_additions = []
    current_removals = []

    for line in diff_text.split("\n"):
        if line.startswith("+++ b/") or line.startswith("+++ "):
            file_path = line.replace("+++ b/", "").replace("+++ ", "").strip()
            current_file = file_path
        elif line.startswith("--- a/") or line.startswith("--- "):
            continue  # Skip the old file header
        elif line.startswith("@@"):
            continue  # Skip hunk headers
        elif current_file is not None:
            if line.startswith("+") and not line.startswith("+++"):
                current_additions.append(line[1:])
            elif line.startswith("-") and not line.startswith("---"):
                current_removals.append(line[1:])

    # If we parsed a diff, try to apply it
    # For simplicity, if the diff is complex, use the LLM to apply it
    if current_file and (current_additions or current_removals):
        if current_file in original_files:
            applied = _apply_diff_simple(
                original_files[current_file],
                current_removals,
                current_additions,
            )
            file_changes[current_file] = applied
        else:
            # File might be new — use additions as content
            file_changes[current_file] = "\n".join(current_additions)
    else:
        # Fallback: use LLM to apply the fix
        file_changes = _apply_fix_with_llm(diff_text, state)

    return file_changes


def _apply_diff_simple(
    original: str, removals: List[str], additions: List[str]
) -> str:
    """
    Simple diff application: replace removed lines with added lines.
    """
    result = original
    for removal in removals:
        result = result.replace(removal.rstrip(), "", 1)

    # Add the new lines where the removals were
    if additions:
        addition_text = "\n".join(additions)
        # Find the best insertion point
        lines = result.split("\n")
        # Remove empty lines that were left by removals
        cleaned_lines = [l for l in lines if l.strip() != "" or l == ""]
        # Insert additions at the point of first removal
        result = "\n".join(cleaned_lines)
        if removals:
            # Try to find the line before the first removal in original
            orig_lines = original.split("\n")
            for i, line in enumerate(orig_lines):
                if removals[0].rstrip() == line.rstrip():
                    # Insert additions at this position
                    cleaned = original.split("\n")
                    # Remove the old lines
                    for rem in removals:
                        for j, cl in enumerate(cleaned):
                            if cl.rstrip() == rem.rstrip():
                                cleaned[j] = None
                                break
                    cleaned = [l for l in cleaned if l is not None]
                    # Insert additions at position i
                    for k, add in enumerate(additions):
                        cleaned.insert(i + k, add)
                    return "\n".join(cleaned)

    return result


def _apply_fix_with_llm(diff_text: str, state: Dict[str, Any]) -> Dict[str, str]:
    """
    Fallback: use Gemini to apply the diff to the original file.
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0,
        )

        relevant_files = state.get("relevant_files", [])
        file_context = ""
        for f in relevant_files:
            fd = f if isinstance(f, dict) else f.dict() if hasattr(f, 'dict') else {}
            path = fd.get("path", "unknown")
            content = fd.get("content", "")
            file_context += f"### {path}\n```\n{content}\n```\n\n"

        user_msg = (
            f"## Original Files\n{file_context}\n"
            f"## Diff to Apply\n```diff\n{diff_text}\n```\n\n"
            f"## Task\n"
            f"Apply the diff to the original files. "
            f"Return a JSON object where keys are file paths and values are the "
            f"COMPLETE updated file content (not just the changes).\n"
            f"Return ONLY the JSON object."
        )

        response = llm.invoke([HumanMessage(content=user_msg)])
        text = response.content.strip()

        # Clean markdown fences
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
        if text.startswith("```"):
            text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

        return json.loads(text)

    except Exception as e:
        logger.error("LLM diff application failed: %s", e)
        # Last resort: return the first relevant file with the diff as a comment
        for f in state.get("relevant_files", []):
            fd = f if isinstance(f, dict) else f.dict() if hasattr(f, 'dict') else {}
            if fd.get("path"):
                return {fd["path"]: fd.get("content", "") + f"\n# FIXME: Apply diff manually\n"}
        return {}


def _determine_test_path(state: Dict[str, Any]) -> str:
    """
    Determine the path for the test file based on the fixed files.
    """
    relevant_files = state.get("relevant_files", [])
    if relevant_files:
        fd = relevant_files[0] if isinstance(relevant_files[0], dict) else relevant_files[0].dict()
        path = fd.get("path", "")
        if path:
            # Create test file next to the source or in tests/
            name = Path(path).stem
            return f"tests/test_{name}.py"
    return f"tests/test_fix_issue_{state.get('issue_number', 0)}.py"


def _generate_pr_content(state: Dict[str, Any]) -> tuple:
    """
    Use Gemini to generate a professional PR title and body.

    Returns:
        (pr_title, pr_body) tuple.
    """
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=AGENT_TEMPERATURES["pr_writer"],
    )

    issue_number = state.get("issue_number", 0)
    exec_result = state.get("execution_result", {})
    if isinstance(exec_result, dict):
        tests_passed = exec_result.get("tests_passed", 0)
    else:
        tests_passed = getattr(exec_result, "tests_passed", 0)

    user_msg = (
        f"## Context\n"
        f"**Issue #{issue_number}:** {state.get('issue_title', '')}\n"
        f"**Issue body:** {state.get('issue_body', '')}\n\n"
        f"**Fix explanation:** {state.get('fix_explanation', '')}\n\n"
        f"**Tests passed:** {tests_passed}\n\n"
        f"## Task\n"
        f"Write a PR title and body. Use this exact format:\n\n"
        f"TITLE: Fix: <concise description>\n\n"
        f"BODY:\n"
        f"## Summary\n<one sentence>\n\n"
        f"## Root Cause\n<what caused the bug>\n\n"
        f"## Fix Approach\n<what changed and why>\n\n"
        f"## Tests Added\n<list each test and what it covers>\n\n"
        f"## Related Issue\nCloses #{issue_number}\n"
    )

    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    text = response.content.strip()

    # Parse title and body
    pr_title = f"Fix: {state.get('issue_title', 'issue')} (#{issue_number})"
    pr_body = text

    if "TITLE:" in text:
        parts = text.split("BODY:", 1)
        title_part = parts[0].replace("TITLE:", "").strip()
        if title_part:
            pr_title = title_part
        if len(parts) > 1:
            pr_body = parts[1].strip()

    return pr_title, pr_body
