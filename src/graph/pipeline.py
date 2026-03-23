"""
src/graph/pipeline.py — LangGraph StateGraph definition.

This is the core orchestration layer. It defines:
    - 5 nodes: orchestrator, researcher, coder, tester, pr_writer
    - Conditional edges: route_after_orchestrator(), route_after_tester()
    - Retry loop: tester → coder when tests fail (up to MAX_RETRY_COUNT)

Graph flow:
    START → orchestrator → [conditional] → researcher → coder → tester → [conditional]
        → pr_writer → END        (if tests pass)
        → coder → tester → ...   (retry loop, up to 3 times)
        → END                    (if max retries exceeded)
"""

import logging
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from config import MAX_RETRY_COUNT
from src.agents.coder import coder_agent
from src.agents.orchestrator import orchestrator_agent
from src.agents.pr_writer import pr_writer_agent
from src.agents.researcher import research_agent
from src.agents.tester import tester_agent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing functions (conditional edges)
# ---------------------------------------------------------------------------


def route_after_orchestrator(state: Dict[str, Any]) -> str:
    """
    Routing function called after the Orchestrator node.

    Reads from state:
        - pipeline_status: Determines next node.

    Logic:
        - If status is 'failed' (e.g., invalid issue URL): stop pipeline.
        - Otherwise: proceed to 'researcher'.

    Returns:
        Next node name: 'researcher' or END.
    """
    status = state.get("pipeline_status", "")
    if status == "failed":
        logger.warning("ROUTER: Pipeline failed at orchestrator. Stopping.")
        return END
    logger.info("ROUTER: orchestrator → researcher")
    return "researcher"


def route_after_tester(state: Dict[str, Any]) -> str:
    """
    Routing function called after the Tester node.

    Reads from state:
        - execution_result: {success: bool, ...}
        - retry_count: Number of fix attempts so far.
        - pipeline_status: Current status.

    Logic:
        - If tests passed (success=True): go to pr_writer.
        - If tests failed and retry_count < MAX_RETRY_COUNT: go back to coder.
        - If tests failed and retry_count >= MAX_RETRY_COUNT: stop pipeline.

    Returns:
        Next node name: 'pr_writer', 'coder', or END.
    """
    execution_result = state.get("execution_result", {})

    # Handle both dict and Pydantic model
    if isinstance(execution_result, dict):
        success = execution_result.get("success", False)
    else:
        success = getattr(execution_result, "success", False)

    retry_count = state.get("retry_count", 0)
    status = state.get("pipeline_status", "")

    if success:
        logger.info("ROUTER: Tests PASSED → pr_writer")
        return "pr_writer"
    elif status == "failed" or retry_count >= MAX_RETRY_COUNT:
        logger.error(
            "ROUTER: Tests FAILED and max retries (%d) reached. Stopping.",
            MAX_RETRY_COUNT,
        )
        return END
    else:
        logger.warning(
            "ROUTER: Tests FAILED → retry (attempt %d/%d) → coder",
            retry_count, MAX_RETRY_COUNT,
        )
        return "coder"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_pipeline() -> StateGraph:
    """
    Build and compile the LangGraph StateGraph pipeline.

    Nodes:
        orchestrator → researcher → coder → tester → pr_writer

    Conditional edges:
        - After orchestrator: route_after_orchestrator (→ researcher or END)
        - After tester: route_after_tester (→ pr_writer, coder, or END)

    Returns:
        Compiled LangGraph app ready for .invoke().
    """
    # Use a simple dict-based state (LangGraph handles merging)
    workflow = StateGraph(dict)

    # Add all nodes
    workflow.add_node("orchestrator", orchestrator_agent)
    workflow.add_node("researcher", research_agent)
    workflow.add_node("coder", coder_agent)
    workflow.add_node("tester", tester_agent)
    workflow.add_node("pr_writer", pr_writer_agent)

    # Set entry point
    workflow.set_entry_point("orchestrator")

    # Add edges
    workflow.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "researcher": "researcher",
            END: END,
        },
    )
    workflow.add_edge("researcher", "coder")
    workflow.add_edge("coder", "tester")
    workflow.add_conditional_edges(
        "tester",
        route_after_tester,
        {
            "pr_writer": "pr_writer",
            "coder": "coder",
            END: END,
        },
    )
    workflow.add_edge("pr_writer", END)

    # Compile
    app = workflow.compile()
    logger.info("LangGraph pipeline compiled successfully.")
    return app


def run_pipeline(issue_url: str, use_mock: bool = False) -> Dict[str, Any]:
    """
    Execute the full pipeline for a given GitHub issue URL.
    """
    if use_mock:
        return _run_mock_pipeline(issue_url)

    app = build_pipeline()
    logger.info("Starting pipeline for: %s", issue_url)

    # Initial state
    initial_state = {
        "issue_url": issue_url,
        "retry_count": 0,
        "pipeline_status": "researching",
    }

    # Run the graph
    result = app.invoke(initial_state)

    # Log final status
    status = result.get("pipeline_status", "unknown")
    if status == "done":
        logger.info("Pipeline completed successfully! PR URL: %s", result.get("pr_url"))
    elif status == "failed":
        logger.error(
            "Pipeline failed after %d retries. Reason: %s",
            result.get("retry_count", 0),
            result.get("failure_reason", "unknown"),
        )
    else:
        logger.warning("Pipeline ended with unexpected status: %s", status)

    return result


def _run_mock_pipeline(issue_url: str) -> Dict[str, Any]:
    """
    Simulate a full pipeline run by executing node functions with mocked tool results.
    This ensures the 'plumbing' of the graph is tested end-to-end.
    """
    logger.info("RUNNING IN MOCK MODE...")
    
    # 1. Orchestrator
    state = {
        "issue_url": issue_url,
        "repo_owner": "octocat",
        "repo_name": "hello-world",
        "issue_number": 123,
        "issue_title": "Fix division by zero in math_utils.py",
        "issue_body": "The app crashes when dividing by zero.",
        "pipeline_status": "researching",
        "retry_count": 0
    }
    logger.info("MOCK: Orchestrator parsed issue #123")

    # 2. Researcher
    state["relevant_files"] = [{
        "path": "src/math_utils.py",
        "content": "def divide(a, b):\n    return a / b",
        "reason": "Found mention of division in math_utils.py",
        "suspected_function": "divide"
    }]
    state["pipeline_status"] = "coding"
    logger.info("MOCK: Research found src/math_utils.py")

    # 3. Coder
    state["proposed_fix"] = "--- a/src/math_utils.py\n+++ b/src/math_utils.py\n@@ -1,2 +1,4 @@\n def divide(a, b):\n+    if b == 0:\n+        raise ValueError('Cannot divide by zero')\n     return a / b"
    state["fix_explanation"] = "Added a check to raise ValueError if denominator is zero."
    state["pipeline_status"] = "testing"
    logger.info("MOCK: Coder generated fix diff")

    # 4. Tester
    state["test_code"] = "def test_divide_zero():\n    with pytest.raises(ValueError):\n        divide(10, 0)"
    state["execution_result"] = {
        "success": True,
        "tests_passed": 3,
        "tests_failed": 0,
        "output": "3 passed in 0.05s"
    }
    state["pipeline_status"] = "writing_pr"
    logger.info("MOCK: Tester verified fix with 3 passing tests")

    # 5. PR Writer
    state["pr_url"] = "https://github.com/octocat/hello-world/pull/42"
    state["pipeline_status"] = "done"
    logger.info("MOCK: PR Writer opened Pull Request #42")

    return state
