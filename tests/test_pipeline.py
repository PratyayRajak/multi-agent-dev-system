"""
tests/test_pipeline.py — Unit tests for the Multi-Agent Dev System.

Tests run WITHOUT real LLM calls or GitHub API — uses mocked state objects
to verify routing logic, state validation, and agent function signatures.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===================================================================
# Test 1: PipelineState model instantiation
# ===================================================================

class TestPipelineState:
    """Test the PipelineState Pydantic model."""

    def test_minimal_state(self):
        """Creating a state with just issue_url should work."""
        from src.state.state import PipelineState
        state = PipelineState(issue_url="https://github.com/owner/repo/issues/1")
        assert state.issue_url == "https://github.com/owner/repo/issues/1"
        assert state.retry_count == 0
        assert state.pipeline_status == "researching"
        assert state.proposed_fix is None
        assert state.pr_url is None

    def test_full_state(self):
        """Creating a state with all fields should work."""
        from src.state.state import PipelineState, RelevantFile, ExecutionResult
        state = PipelineState(
            issue_url="https://github.com/test/repo/issues/42",
            issue_number=42,
            issue_title="Bug in division",
            issue_body="divide_numbers(10, 0) crashes",
            repo_owner="test",
            repo_name="repo",
            relevant_files=[
                RelevantFile(
                    path="src/math.py",
                    content="def divide(a, b): return a/b",
                    reason="Contains divide function",
                    suspected_function="divide",
                )
            ],
            proposed_fix="--- a/src/math.py\n+++ b/src/math.py",
            fix_explanation="Added zero check",
            test_code="def test_divide(): assert divide(10, 2) == 5",
            execution_result=ExecutionResult(
                success=True, output="1 passed", tests_passed=1, tests_failed=0
            ),
            retry_count=0,
            pr_url="https://github.com/test/repo/pull/1",
            pipeline_status="done",
        )
        assert state.issue_number == 42
        assert len(state.relevant_files) == 1
        assert state.execution_result.success is True

    def test_execution_result(self):
        """ExecutionResult model should validate correctly."""
        from src.state.state import ExecutionResult
        result = ExecutionResult(
            success=False,
            output="FAILED test_divide",
            error="ZeroDivisionError",
            tests_passed=2,
            tests_failed=1,
        )
        assert result.success is False
        assert result.tests_failed == 1


# ===================================================================
# Test 2: Orchestrator — URL parsing
# ===================================================================

class TestOrchestrator:
    """Test the Orchestrator Agent's URL parsing."""

    def test_parse_valid_url(self):
        """Should correctly extract owner, repo, and issue number."""
        from src.agents.orchestrator import parse_issue_url
        result = parse_issue_url("https://github.com/octocat/hello-world/issues/42")
        assert result["owner"] == "octocat"
        assert result["repo"] == "hello-world"
        assert result["issue_number"] == 42

    def test_parse_url_with_trailing_slash(self):
        """Should handle URLs with trailing content."""
        from src.agents.orchestrator import parse_issue_url
        result = parse_issue_url("https://github.com/user/repo/issues/1")
        assert result["owner"] == "user"
        assert result["repo"] == "repo"
        assert result["issue_number"] == 1

    def test_parse_invalid_url(self):
        """Should raise ValueError for invalid URLs."""
        from src.agents.orchestrator import parse_issue_url
        with pytest.raises(ValueError, match="Invalid GitHub issue URL"):
            parse_issue_url("https://google.com/not-a-github-url")

    def test_parse_url_no_issue_number(self):
        """Should raise ValueError if no issue number."""
        from src.agents.orchestrator import parse_issue_url
        with pytest.raises(ValueError):
            parse_issue_url("https://github.com/owner/repo/issues/")


# ===================================================================
# Test 3: Routing — route_after_orchestrator
# ===================================================================

class TestRoutingOrchestrator:
    """Test the route_after_orchestrator conditional routing."""

    def test_route_to_researcher_on_success(self):
        """When pipeline_status is 'researching', should route to researcher."""
        from src.graph.pipeline import route_after_orchestrator
        state = {"pipeline_status": "researching"}
        assert route_after_orchestrator(state) == "researcher"

    def test_route_to_end_on_failure(self):
        """When pipeline_status is 'failed', should stop pipeline."""
        from src.graph.pipeline import route_after_orchestrator
        from langgraph.graph import END
        state = {"pipeline_status": "failed"}
        assert route_after_orchestrator(state) == END


# ===================================================================
# Test 4: Routing — route_after_tester
# ===================================================================

class TestRoutingTester:
    """Test the route_after_tester conditional routing."""

    def test_route_to_pr_writer_on_pass(self):
        """When tests pass, should route to pr_writer."""
        from src.graph.pipeline import route_after_tester
        state = {
            "execution_result": {"success": True},
            "retry_count": 0,
            "pipeline_status": "writing_pr",
        }
        assert route_after_tester(state) == "pr_writer"

    def test_route_to_coder_on_fail_under_max_retries(self):
        """When tests fail and retries remain, should retry via coder."""
        from src.graph.pipeline import route_after_tester
        state = {
            "execution_result": {"success": False},
            "retry_count": 1,
            "pipeline_status": "retrying",
        }
        assert route_after_tester(state) == "coder"

    def test_route_to_end_on_max_retries(self):
        """When tests fail and max retries reached, should stop."""
        from src.graph.pipeline import route_after_tester
        from langgraph.graph import END
        state = {
            "execution_result": {"success": False},
            "retry_count": 3,
            "pipeline_status": "failed",
        }
        assert route_after_tester(state) == END

    def test_route_to_end_on_explicit_failure(self):
        """When pipeline_status is 'failed', should stop regardless of count."""
        from src.graph.pipeline import route_after_tester
        from langgraph.graph import END
        state = {
            "execution_result": {"success": False},
            "retry_count": 0,
            "pipeline_status": "failed",
        }
        assert route_after_tester(state) == END


# ===================================================================
# Test 5: Docker sandbox output parsing
# ===================================================================

class TestDockerOutputParsing:
    """Test pytest output parsing in the Docker runner."""

    def test_parse_all_passed(self):
        """Should correctly parse when all tests pass."""
        from src.sandbox.docker_runner import _parse_pytest_output
        output = "====== 5 passed in 0.12s ======"
        result = _parse_pytest_output(output)
        assert result["success"] is True
        assert result["tests_passed"] == 5
        assert result["tests_failed"] == 0

    def test_parse_mixed_results(self):
        """Should correctly parse mixed pass/fail output."""
        from src.sandbox.docker_runner import _parse_pytest_output
        output = "====== 3 passed, 2 failed in 0.45s ======"
        result = _parse_pytest_output(output)
        assert result["success"] is False
        assert result["tests_passed"] == 3
        assert result["tests_failed"] == 2

    def test_parse_all_failed(self):
        """Should correctly parse when all tests fail."""
        from src.sandbox.docker_runner import _parse_pytest_output
        output = "====== 4 failed in 0.33s ======"
        result = _parse_pytest_output(output)
        assert result["success"] is False
        assert result["tests_passed"] == 0
        assert result["tests_failed"] == 4

    def test_parse_with_errors(self):
        """Should detect errors in test output."""
        from src.sandbox.docker_runner import _parse_pytest_output
        output = "====== 1 passed, 1 error in 0.10s ======"
        result = _parse_pytest_output(output)
        assert result["success"] is False

    def test_parse_empty_output(self):
        """Should handle empty output gracefully."""
        from src.sandbox.docker_runner import _parse_pytest_output
        result = _parse_pytest_output("")
        assert result["success"] is False
        assert result["tests_passed"] == 0


# ===================================================================
# Test 6: Config health checks
# ===================================================================

class TestConfig:
    """Test config module health checks."""

    def test_check_google_api_key_missing(self):
        """Should return False when GOOGLE_API_KEY is not set."""
        from config import check_google_api_key
        with patch.dict(os.environ, {}, clear=True):
            # Remove if present
            os.environ.pop("GOOGLE_API_KEY", None)
            assert check_google_api_key() is False

    def test_check_github_token_missing(self):
        """Should return False when GITHUB_TOKEN is not set."""
        from config import check_github_token
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GITHUB_TOKEN", None)
            assert check_github_token() is False
