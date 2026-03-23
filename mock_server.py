"""
mock_server.py — Simulates the Multi-Agent Dev System API for UI demonstration.
Mocks health checks (Green) and a full pipeline run (researching -> coding -> testing -> writing_pr -> done).
"""

import time
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RunRequest(BaseModel):
    issue_url: str

# In-memory store
runs = {}

@app.post("/run")
async def start_run(request: RunRequest):
    run_id = str(uuid.uuid4())
    runs[run_id] = {
        "status": "researching",
        "retry_count": 0,
        "start_time": time.time()
    }
    return {"run_id": run_id, "status": "started"}

@app.get("/status/{run_id}")
async def get_status(run_id: str):
    if run_id not in runs:
        raise HTTPException(status_code=404)
    
    run = runs[run_id]
    elapsed = time.time() - run["start_time"]
    
    # Simulate status transitions
    if elapsed < 5:
        run["status"] = "researching"
    elif elapsed < 12:
        run["status"] = "coding"
    elif elapsed < 20:
        run["status"] = "testing"
    elif elapsed < 28:
        run["status"] = "writing_pr"
    else:
        run["status"] = "done"
        run["pr_url"] = "https://github.com/octocat/hello-world/pull/123"
    
    return {
        "status": run["status"],
        "retry_count": run.get("retry_count", 0),
        "pr_url": run.get("pr_url"),
        "failure_reason": None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "google_api_key": True,
        "github_token": True,
        "docker": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
