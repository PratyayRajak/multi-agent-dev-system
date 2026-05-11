"""
config.py — All constants for the Multi-Agent Software Engineering System.
Every module imports from here. Never hardcode values elsewhere.
Values in config.yaml override the defaults defined here.
"""

import logging
import os
import subprocess
import yaml

# ---------------------------------------------------------------------------
# Load config.yaml (runtime overrides)
# ---------------------------------------------------------------------------
_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
_cfg = {}
if os.path.exists(_config_path):
    with open(_config_path, "r") as f:
        _cfg = yaml.safe_load(f) or {}

def _get(keys: str, default):
    """Dot-notation read from loaded config. e.g. _get('llm.model', 'gemini-2.0-flash')"""
    val = _cfg
    for key in keys.split("."):
        if isinstance(val, dict):
            val = val.get(key)
        else:
            return default
    return val if val is not None else default

# ---------------------------------------------------------------------------
# LLM Configuration (Gemini)
# ---------------------------------------------------------------------------
GEMINI_MODEL = _get("llm.model", "gemini-2.0-flash")

AGENT_TEMPERATURES = {
    "orchestrator": 0.0,
    "researcher": _get("llm.temperature.researcher", 0.0),
    "coder":       _get("llm.temperature.coder", 0.1),
    "tester":      _get("llm.temperature.tester", 0.0),
    "pr_writer":   _get("llm.temperature.pr_writer", 0.2),
}

# ---------------------------------------------------------------------------
# Pipeline Behaviour
# ---------------------------------------------------------------------------
MAX_RETRY_COUNT  = _get("pipeline.max_retries", 3)
MAX_FILES_TO_READ = 10        # Max files the Research Agent reads per run
MOCK_MODE        = _get("pipeline.mock_mode", False)
PIPELINE_TIMEOUT = _get("pipeline.timeout_seconds", 300)

# Valid pipeline status values
PIPELINE_STATUSES = {
    "RESEARCHING": "researching",
    "CODING": "coding",
    "TESTING": "testing",
    "RETRYING": "retrying",
    "WRITING_PR": "writing_pr",
    "DONE": "done",
    "FAILED": "failed",
}

# ---------------------------------------------------------------------------
# Docker Sandbox
# ---------------------------------------------------------------------------
DOCKER_IMAGE_NAME       = _get("docker.image", "rag-sandbox")
DOCKER_DOCKERFILE       = "docker/Dockerfile.sandbox"
DOCKER_MEMORY_LIMIT     = _get("docker.memory_limit", "256m")
DOCKER_CPU_LIMIT        = str(_get("docker.cpu_limit", 0.5))
DOCKER_TIMEOUT_SECONDS  = _get("docker.timeout_seconds", 30)
DOCKER_WORKSPACE_DIR    = "/workspace"

# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------
GITHUB_BASE_BRANCH              = "main"
GITHUB_FIX_BRANCH_PREFIX        = "fix/issue-"
GITHUB_RATE_LIMIT_BACKOFF_SECONDS = 60

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}
LOG_LEVEL  = _log_level_map.get(_get("logging.level", "INFO"), logging.INFO)
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_TO_FILE = _get("logging.log_to_file", False)
LOG_FILE    = _get("logging.log_file", "runs.log")

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("config")

# ---------------------------------------------------------------------------
# LangSmith Observability (optional)
# ---------------------------------------------------------------------------
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "multi-agent-dev-system")


# ---------------------------------------------------------------------------
# Health Checks
# ---------------------------------------------------------------------------

def check_google_api_key() -> bool:
    """Verify GOOGLE_API_KEY is set in environment."""
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        logger.error(
            "GOOGLE_API_KEY is not set. "
            "Get a key from https://aistudio.google.com/app/apikey"
        )
        return False
    return True


def check_github_token() -> bool:
    """Verify GITHUB_TOKEN is set in environment."""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        logger.error(
            "GITHUB_TOKEN is not set. "
            "Create a token at https://github.com/settings/tokens "
            "with scopes: repo (read+write) and pull_requests (write)."
        )
        return False
    return True


def check_docker_available() -> bool:
    """Verify Docker daemon is running and accessible."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            logger.error(
                "Docker is not running. Start Docker Desktop and retry. "
                f"Docker error: {result.stderr[:200]}"
            )
            return False
        return True
    except FileNotFoundError:
        logger.error(
            "Docker executable not found. Install Docker Desktop from https://www.docker.com/products/docker-desktop"
        )
        return False
    except subprocess.TimeoutExpired:
        logger.error("Docker health check timed out after 5 seconds.")
        return False


def run_all_health_checks() -> dict:
    """Run all health checks and return a status dict."""
    results = {
        "google_api_key": check_google_api_key(),
        "github_token": check_github_token(),
        "docker": check_docker_available(),
    }
    all_ok = all(results.values())
    if all_ok:
        logger.info("All health checks passed.")
    else:
        failed = [k for k, v in results.items() if not v]
        logger.warning(f"Health check failures: {failed}")
    return results
