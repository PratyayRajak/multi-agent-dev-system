"""
src/api/main.py — FastAPI REST API for triggering and monitoring the pipeline.

Endpoints:
    POST /run          → Start a new pipeline run for a GitHub issue
    GET  /status/{id}  → Poll the status of a running pipeline
    GET  /health       → Health check (Docker, GitHub, API key)

Runs:
    uvicorn src.api.main:app --reload --port 8000
"""

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import run_all_health_checks
from src.graph.pipeline import run_pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Multi-Agent Dev System API",
    description=(
        "REST API for the Multi-Agent Software Engineering System. "
        "Submit a GitHub issue URL, the system autonomously researches, "
        "fixes, tests, and opens a pull request."
    ),
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory run store (production would use Redis or a database)
# ---------------------------------------------------------------------------

_runs: Dict[str, Dict] = {}
_executor = ThreadPoolExecutor(max_workers=3)


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    """Request body for POST /run."""
    issue_url: str = Field(
        ...,
        description="Full GitHub issue URL, e.g. https://github.com/owner/repo/issues/42",
        examples=["https://github.com/octocat/hello-world/issues/1"],
    )


class RunResponse(BaseModel):
    """Response body for POST /run."""
    run_id: str = Field(..., description="UUID to poll status with")
    status: str = Field(default="started", description="Initial status")


class StatusResponse(BaseModel):
    """Response body for GET /status/{run_id}."""
    status: str = Field(..., description="Pipeline status")
    retry_count: int = Field(default=0, description="Number of fix retries")
    pr_url: Optional[str] = Field(None, description="PR URL when done")
    failure_reason: Optional[str] = Field(None, description="Failure reason when failed")


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str = Field(..., description="Overall health status")
    google_api_key: bool = Field(..., description="Google API key configured")
    github_token: bool = Field(..., description="GitHub token configured")
    docker: bool = Field(..., description="Docker available")


# ---------------------------------------------------------------------------
# Background task runner
# ---------------------------------------------------------------------------


def _run_pipeline_async(run_id: str, issue_url: str) -> None:
    """
    Run the pipeline in a background thread and update the run store.
    """
    try:
        logger.info("Background run %s started for %s", run_id, issue_url)
        result = run_pipeline(issue_url)
        _runs[run_id] = {
            "status": result.get("pipeline_status", "unknown"),
            "retry_count": result.get("retry_count", 0),
            "pr_url": result.get("pr_url"),
            "failure_reason": result.get("failure_reason"),
        }
        logger.info("Background run %s completed: %s", run_id, _runs[run_id]["status"])
    except Exception as e:
        logger.error("Background run %s failed: %s", run_id, e)
        _runs[run_id] = {
            "status": "failed",
            "retry_count": 0,
            "pr_url": None,
            "failure_reason": str(e),
        }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/run", response_model=RunResponse)
async def start_run(request: RunRequest):
    """
    Start a new pipeline run for a GitHub issue.

    Triggers the full multi-agent pipeline in a background thread.
    Returns a run_id to poll for status.
    """
    run_id = str(uuid.uuid4())
    _runs[run_id] = {
        "status": "started",
        "retry_count": 0,
        "pr_url": None,
        "failure_reason": None,
    }

    # Start pipeline in background
    _executor.submit(_run_pipeline_async, run_id, request.issue_url)

    logger.info("Started run %s for issue: %s", run_id, request.issue_url)
    return RunResponse(run_id=run_id, status="started")


@app.get("/status/{run_id}", response_model=StatusResponse)
async def get_status(run_id: str):
    """
    Poll the status of a pipeline run.

    Returns current status, retry count, PR URL (if done), and failure reason (if failed).
    """
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    run_data = _runs[run_id]
    return StatusResponse(**run_data)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    System health check.

    Verifies:
    - Google API key is configured
    - GitHub token is configured
    - Docker daemon is accessible
    """
    checks = run_all_health_checks()
    all_ok = all(checks.values())
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        google_api_key=checks.get("google_api_key", False),
        github_token=checks.get("github_token", False),
        docker=checks.get("docker", False),
    )
