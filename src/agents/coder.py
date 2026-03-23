"""
src/agents/coder.py — Coder Agent.

Reads:  state.relevant_files, state.issue_title, state.issue_body,
        state.failure_reason (on retry), state.retry_count
Writes: state.proposed_fix (unified diff), state.fix_explanation,
        state.pipeline_status = 'testing'

Role:
    - Reads relevant codebase files from state.
    - On retry, reads failure_reason to understand what went wrong.
    - Writes a targeted, minimal fix in unified diff format.
    - Explains the fix clearly.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import AGENT_TEMPERATURES, GEMINI_MODEL

logger = logging.getLogger(__name__)

# Load system prompt
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "coder.txt"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


def coder_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coder Agent node for the LangGraph pipeline.

    Reads from state:
        - relevant_files: Files identified by the Research Agent.
        - issue_title, issue_body: Bug description.
        - failure_reason: Why the previous fix failed (if retry).
        - retry_count: Current retry attempt number.

    Writes to state:
        - proposed_fix: The code fix in unified diff format.
        - fix_explanation: Plain English explanation.
        - pipeline_status: Set to 'testing'.

    Args:
        state: Current pipeline state dict.

    Returns:
        Updated state dict.
    """
    retry_count = state.get("retry_count", 0)
    failure_reason = state.get("failure_reason")
    is_retry = retry_count > 0 and failure_reason

    logger.info("=" * 60)
    if is_retry:
        logger.info(
            "CODER: Retry attempt %d — fixing based on failure feedback", retry_count
        )
        logger.info("Failure reason: %s", failure_reason)
    else:
        logger.info("CODER: Writing initial fix")
    logger.info("=" * 60)

    # Build context for LLM
    user_message = _build_user_message(state, is_retry)

    # Call Gemini
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=AGENT_TEMPERATURES["coder"],
    )

    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    response_text = response.content.strip()

    # Parse the response — extract proposed_fix and fix_explanation
    proposed_fix, fix_explanation = _parse_response(response_text)

    logger.info("CODER: Fix generated (%d chars)", len(proposed_fix))
    logger.info("CODER: Explanation: %s", fix_explanation[:200])

    return {
        "proposed_fix": proposed_fix,
        "fix_explanation": fix_explanation,
        "pipeline_status": "testing",
    }


def _build_user_message(state: Dict[str, Any], is_retry: bool) -> str:
    """Build the user message with all context for the Coder Agent."""
    parts = []

    # Issue context
    parts.append(f"## Issue")
    parts.append(f"**Title:** {state.get('issue_title', '')}")
    parts.append(f"**Body:** {state.get('issue_body', '')}")
    parts.append("")

    # Relevant files
    relevant_files = state.get("relevant_files", [])
    if relevant_files:
        parts.append("## Relevant Files")
        for f in relevant_files:
            file_data = f if isinstance(f, dict) else f.dict() if hasattr(f, 'dict') else {}
            path = file_data.get("path", "unknown")
            content = file_data.get("content", "")
            reason = file_data.get("reason", "")
            suspected = file_data.get("suspected_function", "")
            parts.append(f"### {path}")
            parts.append(f"**Reason:** {reason}")
            if suspected:
                parts.append(f"**Suspected function:** {suspected}")
            parts.append(f"```\n{content}\n```")
            parts.append("")

    # Retry context
    if is_retry:
        parts.append("## RETRY — Previous Fix Failed")
        parts.append(f"**Attempt:** {state.get('retry_count', 0)}")
        parts.append(f"**Failure reason:** {state.get('failure_reason', '')}")
        parts.append("")
        if state.get("proposed_fix"):
            parts.append("**Previous fix (that failed):**")
            parts.append(f"```diff\n{state['proposed_fix']}\n```")
            parts.append("")
        parts.append(
            "Address the SPECIFIC failure described above. "
            "Do not start from scratch — refine the previous fix."
        )
    else:
        parts.append("## Task")
        parts.append(
            "Write a minimal, targeted fix for this bug. "
            "Output the fix in unified diff format and explain your changes."
        )

    parts.append("")
    parts.append(
        "## Output Format\n"
        "Return your response in this exact format:\n\n"
        "### PROPOSED_FIX\n"
        "```diff\n"
        "your unified diff here\n"
        "```\n\n"
        "### FIX_EXPLANATION\n"
        "Your explanation here."
    )

    return "\n".join(parts)


def _parse_response(response_text: str) -> tuple:
    """
    Parse the LLM response to extract proposed_fix and fix_explanation.

    Returns:
        (proposed_fix, fix_explanation) tuple of strings.
    """
    proposed_fix = ""
    fix_explanation = ""

    # Try to extract diff block
    if "```diff" in response_text:
        parts = response_text.split("```diff")
        if len(parts) > 1:
            diff_block = parts[1].split("```")[0]
            proposed_fix = diff_block.strip()
    elif "```" in response_text:
        # Fall back to any code block
        parts = response_text.split("```")
        if len(parts) > 1:
            proposed_fix = parts[1].strip()
            # Remove language identifier if present
            if proposed_fix and "\n" in proposed_fix:
                first_line = proposed_fix.split("\n")[0]
                if first_line.strip() in ("python", "diff", "py", "javascript", "js"):
                    proposed_fix = "\n".join(proposed_fix.split("\n")[1:])

    # Try to extract explanation
    if "### FIX_EXPLANATION" in response_text:
        explanation_part = response_text.split("### FIX_EXPLANATION")[1]
        fix_explanation = explanation_part.strip()
    elif "FIX_EXPLANATION" in response_text:
        explanation_part = response_text.split("FIX_EXPLANATION")[1]
        fix_explanation = explanation_part.strip().lstrip(":").strip()
    else:
        # Use everything after the diff block as explanation
        if "```" in response_text:
            parts = response_text.split("```")
            if len(parts) > 2:
                fix_explanation = parts[-1].strip()
            else:
                fix_explanation = "Fix applied as shown in the diff."
        else:
            fix_explanation = response_text

    # Fallback if no diff was found
    if not proposed_fix:
        proposed_fix = response_text
        fix_explanation = "Raw LLM response — could not parse structured output."

    return proposed_fix, fix_explanation
