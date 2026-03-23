"""
src/agents/tester.py — Tester Agent.

Reads:  state.proposed_fix, state.relevant_files, state.issue_title,
        state.issue_body, state.fix_explanation
Writes: state.test_code, state.execution_result, state.failure_reason (on fail),
        state.retry_count (incremented on fail), state.pipeline_status

Role:
    - Writes pytest tests targeting the bug described in the issue.
    - Runs those tests inside the Docker sandbox against the proposed fix.
    - On pass: sets pipeline_status = 'writing_pr'.
    - On fail: writes failure_reason, increments retry_count, sets status = 'retrying'.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import AGENT_TEMPERATURES, GEMINI_MODEL, MAX_RETRY_COUNT
from src.sandbox.docker_runner import run_tests_in_sandbox

logger = logging.getLogger(__name__)

# Load system prompt
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "tester.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


def tester_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tester Agent node for the LangGraph pipeline.

    Reads from state:
        - proposed_fix: The code fix (unified diff) from the Coder Agent.
        - relevant_files: Files context from the Research Agent.
        - issue_title, issue_body: Bug description.
        - fix_explanation: What the Coder Agent changed and why.

    Writes to state:
        - test_code: Full pytest test file content.
        - execution_result: {success, output, error, tests_passed, tests_failed}
        - failure_reason: Why tests failed (only on failure).
        - retry_count: Incremented on failure.
        - pipeline_status: 'writing_pr' on success, 'retrying' on failure.

    Args:
        state: Current pipeline state dict.

    Returns:
        Updated state dict.
    """
    retry_count = state.get("retry_count", 0)
    logger.info("=" * 60)
    logger.info("TESTER: Writing tests and validating fix (attempt %d)", retry_count + 1)
    logger.info("=" * 60)

    # Step 1: Generate test code using Gemini
    test_code = _generate_tests(state)
    logger.info("TESTER: Generated test code (%d chars)", len(test_code))

    # Step 2: Prepare source files with the fix applied
    source_files = _prepare_fixed_source(state)

    # Step 3: Run tests in Docker sandbox
    logger.info("TESTER: Running tests in Docker sandbox...")
    try:
        execution_result = run_tests_in_sandbox(
            fix_code=state.get("proposed_fix", ""),
            test_code=test_code,
            relevant_files=state.get("relevant_files", []),
        )
    except Exception as e:
        logger.error("TESTER: Docker sandbox execution failed: %s", e)
        execution_result = {
            "success": False,
            "output": "",
            "error": str(e),
            "tests_passed": 0,
            "tests_failed": 0,
        }

    success = execution_result.get("success", False)

    logger.info(
        "TESTER: Result — %s (passed=%d, failed=%d)",
        "PASS" if success else "FAIL",
        execution_result.get("tests_passed", 0),
        execution_result.get("tests_failed", 0),
    )

    # Step 4: Determine next state
    if success:
        logger.info("TESTER: All tests passed! Proceeding to PR Writer.")
        return {
            "test_code": test_code,
            "execution_result": execution_result,
            "failure_reason": None,
            "pipeline_status": "writing_pr",
        }
    else:
        # Generate specific failure analysis
        failure_reason = _analyze_failure(state, test_code, execution_result)
        new_retry_count = retry_count + 1

        if new_retry_count >= MAX_RETRY_COUNT:
            logger.error(
                "TESTER: Max retries (%d) reached. Pipeline failed.", MAX_RETRY_COUNT
            )
            return {
                "test_code": test_code,
                "execution_result": execution_result,
                "failure_reason": failure_reason,
                "retry_count": new_retry_count,
                "pipeline_status": "failed",
            }
        else:
            logger.warning(
                "TESTER: Tests failed. Retry %d/%d — sending feedback to Coder.",
                new_retry_count, MAX_RETRY_COUNT,
            )
            return {
                "test_code": test_code,
                "execution_result": execution_result,
                "failure_reason": failure_reason,
                "retry_count": new_retry_count,
                "pipeline_status": "retrying",
            }


def _generate_tests(state: Dict[str, Any]) -> str:
    """
    Use Gemini to generate pytest tests for the proposed fix.
    """
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=AGENT_TEMPERATURES["tester"],
    )

    # Build context
    relevant_files = state.get("relevant_files", [])
    file_context = ""
    for f in relevant_files:
        fd = f if isinstance(f, dict) else f.dict() if hasattr(f, 'dict') else {}
        path = fd.get("path", "unknown")
        content = fd.get("content", "")
        file_context += f"### {path}\n```python\n{content}\n```\n\n"

    user_msg = (
        f"## Issue\n"
        f"**Title:** {state.get('issue_title', '')}\n"
        f"**Body:** {state.get('issue_body', '')}\n\n"
        f"## Relevant Files\n{file_context}\n"
        f"## Proposed Fix\n```diff\n{state.get('proposed_fix', '')}\n```\n\n"
        f"## Fix Explanation\n{state.get('fix_explanation', '')}\n\n"
        f"## Task\n"
        f"Write a complete pytest test file. Include at minimum:\n"
        f"1. A happy path test\n"
        f"2. A boundary condition test (the exact input that triggers the bug)\n"
        f"3. An edge case test\n\n"
        f"Return ONLY the full Python test file content, nothing else. "
        f"Do not wrap it in markdown code fences."
    )

    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    test_code = response.content.strip()

    # Clean up markdown fences if present
    if test_code.startswith("```python"):
        test_code = test_code[len("```python"):].strip()
    if test_code.startswith("```"):
        test_code = test_code[3:].strip()
    if test_code.endswith("```"):
        test_code = test_code[:-3].strip()

    return test_code


def _prepare_fixed_source(state: Dict[str, Any]) -> Dict[str, str]:
    """
    Prepare source file contents with the proposed fix applied.
    Returns a dict of {file_path: fixed_content}.
    """
    files = {}
    for f in state.get("relevant_files", []):
        fd = f if isinstance(f, dict) else f.dict() if hasattr(f, 'dict') else {}
        path = fd.get("path", "")
        content = fd.get("content", "")
        if path and content:
            files[path] = content
    return files


def _analyze_failure(
    state: Dict[str, Any], test_code: str, execution_result: Dict
) -> str:
    """
    Use Gemini to analyze test failures and produce a specific failure_reason
    the Coder Agent can act on.
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0,
        )

        user_msg = (
            f"## Test Execution Failed\n\n"
            f"**Test output (stdout):**\n```\n{execution_result.get('output', '')}\n```\n\n"
            f"**Error (stderr):**\n```\n{execution_result.get('error', '')}\n```\n\n"
            f"**Test code:**\n```python\n{test_code}\n```\n\n"
            f"**Proposed fix:**\n```diff\n{state.get('proposed_fix', '')}\n```\n\n"
            f"## Task\n"
            f"Analyze why the tests failed. Write a specific, actionable failure reason "
            f"that tells the Coder Agent exactly what to fix. Be precise:\n"
            f"- Which test(s) failed\n"
            f"- What the expected vs actual output was\n"
            f"- What the Coder Agent should change\n\n"
            f"Return ONLY the failure reason text, no markdown formatting."
        )

        response = llm.invoke([HumanMessage(content=user_msg)])
        return response.content.strip()

    except Exception as e:
        logger.warning("LLM failure analysis failed: %s", e)
        error_text = execution_result.get("error", "")
        output_text = execution_result.get("output", "")
        return (
            f"Tests failed. Error: {error_text[:500]}. "
            f"Output: {output_text[:500]}"
        )
