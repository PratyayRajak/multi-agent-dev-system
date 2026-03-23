"""
src/state/state.py — Shared PipelineState Pydantic model.

This is the "shared whiteboard" that every agent reads from and writes to.
All agents have the signature: agent_name(state: PipelineState) -> PipelineState

Fields:
  - Inputs set by orchestrator
  - Outputs written by each specialist agent
  - Control fields (retry_count, pipeline_status)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RelevantFile(BaseModel):
    """A codebase file identified as relevant to the issue by the Research Agent."""

    path: str = Field(..., description="File path within the repository, e.g. 'src/math_utils.py'")
    content: str = Field(..., description="Full raw content of the file")
    reason: str = Field(..., description="Why this file is relevant to the issue")
    suspected_function: Optional[str] = Field(
        None, description="Name of function/class suspected to contain the bug"
    )


class ExecutionResult(BaseModel):
    """Result of running pytest inside the Docker sandbox."""

    success: bool = Field(..., description="True if all tests passed")
    output: str = Field(default="", description="Full pytest stdout")
    error: str = Field(default="", description="Full pytest stderr / error message")
    tests_passed: int = Field(default=0, description="Number of tests that passed")
    tests_failed: int = Field(default=0, description="Number of tests that failed")


class PipelineState(BaseModel):
    """
    Shared state object passed through the entire LangGraph pipeline.
    Every agent reads from and writes to this object.

    Reads:
        orchestrator  → issue_url (set by caller)
        researcher    → issue_title, issue_body, repo_owner, repo_name
        coder         → relevant_files, failure_reason (on retry)
        tester        → proposed_fix, relevant_files
        pr_writer     → all fields

    Writes:
        orchestrator  → issue_title, issue_body, repo_owner, repo_name, issue_number, pipeline_status
        researcher    → relevant_files
        coder         → proposed_fix, fix_explanation
        tester        → test_code, execution_result, failure_reason
        pr_writer     → pr_url, pipeline_status = 'done'
    """

    # -----------------------------------------------------------------------
    # Input — set by caller on pipeline start
    # -----------------------------------------------------------------------
    issue_url: str = Field(..., description="Full GitHub issue URL, e.g. https://github.com/owner/repo/issues/1")

    # -----------------------------------------------------------------------
    # Parsed by Orchestrator Agent
    # -----------------------------------------------------------------------
    issue_number: Optional[int] = Field(None, description="Parsed issue number from URL")
    issue_title: Optional[str] = Field(None, description="GitHub issue title")
    issue_body: Optional[str] = Field(None, description="Full issue description text")
    issue_comments: Optional[List[str]] = Field(None, description="All issue comments")
    repo_owner: Optional[str] = Field(None, description="GitHub repository owner username")
    repo_name: Optional[str] = Field(None, description="GitHub repository name (without owner)")

    # -----------------------------------------------------------------------
    # Research Agent Output
    # -----------------------------------------------------------------------
    relevant_files: Optional[List[RelevantFile]] = Field(
        None, description="Files identified as relevant to the issue (max 10)"
    )

    # -----------------------------------------------------------------------
    # Coder Agent Output
    # -----------------------------------------------------------------------
    proposed_fix: Optional[str] = Field(
        None, description="The code fix in unified diff format (--- a/file  +++ b/file  @@ ... @@)"
    )
    fix_explanation: Optional[str] = Field(
        None, description="Plain English explanation of what changed and why"
    )

    # -----------------------------------------------------------------------
    # Tester Agent Output
    # -----------------------------------------------------------------------
    test_code: Optional[str] = Field(
        None, description="Full pytest test file content"
    )
    execution_result: Optional[ExecutionResult] = Field(
        None, description="Result of running tests in Docker sandbox"
    )

    # -----------------------------------------------------------------------
    # Retry Control
    # -----------------------------------------------------------------------
    retry_count: int = Field(
        default=0, description="Number of fix attempts so far (max = MAX_RETRY_COUNT in config)"
    )
    failure_reason: Optional[str] = Field(
        None,
        description="Specific description of what failed — fed back to Coder Agent on retry"
    )

    # -----------------------------------------------------------------------
    # PR Writer Output
    # -----------------------------------------------------------------------
    pr_url: Optional[str] = Field(None, description="URL of the opened pull request")

    # -----------------------------------------------------------------------
    # Pipeline Control
    # -----------------------------------------------------------------------
    pipeline_status: str = Field(
        default="researching",
        description=(
            "Current pipeline status. One of: "
            "researching | coding | testing | retrying | writing_pr | done | failed"
        ),
    )

    # -----------------------------------------------------------------------
    # Internal — misc metadata
    # -----------------------------------------------------------------------
    extra: Optional[Dict[str, Any]] = Field(
        None, description="Optional bag for extra metadata any agent may need"
    )

    class Config:
        # Allow mutation so agents can update state
        validate_assignment = True
