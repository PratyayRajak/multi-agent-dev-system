"""
src/sandbox/docker_runner.py — Docker sandbox for safe code execution.

Runs AI-generated code (fix + tests) inside an isolated Docker container
with no network access, limited memory, and auto-cleanup.

Security flags:
    --network none   → No internet access
    --memory 256m    → Max 256MB RAM
    --cpus 0.5       → Max half a CPU core
    --rm             → Container auto-deleted after run
"""

import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import docker
from docker.errors import ContainerError, DockerException, ImageNotFound

from config import (
    DOCKER_CPU_LIMIT,
    DOCKER_IMAGE_NAME,
    DOCKER_MEMORY_LIMIT,
    DOCKER_TIMEOUT_SECONDS,
    DOCKER_WORKSPACE_DIR,
)

logger = logging.getLogger(__name__)


def run_tests_in_sandbox(
    fix_code: str,
    test_code: str,
    relevant_files: Optional[List[Dict[str, Any]]] = None,
    requirements: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run pytest tests in an isolated Docker container.

    This function:
        1. Writes fix_code and test_code to a temp directory.
        2. Mounts the temp directory into a Docker container.
        3. Runs pytest inside the container.
        4. Captures stdout/stderr.
        5. Parses results.
        6. Cleans up temp directory.

    Reads from state (via args):
        - fix_code: The proposed code fix (unified diff or full file).
        - test_code: The pytest test file content.
        - relevant_files: Original files to include in the sandbox.

    Args:
        fix_code:       The code fix (may be diff or full file content).
        test_code:      Full pytest test file content.
        relevant_files: List of file dicts [{path, content, ...}].
        requirements:   Extra pip packages to install (optional).

    Returns:
        dict: {success, output, error, tests_passed, tests_failed}
    """
    temp_dir = None

    try:
        # Step 1: Create temp directory structure
        temp_dir = tempfile.mkdtemp(prefix="agent_sandbox_")
        workspace = Path(temp_dir)
        src_dir = workspace / "src"
        tests_dir = workspace / "tests"
        src_dir.mkdir(parents=True, exist_ok=True)
        tests_dir.mkdir(parents=True, exist_ok=True)

        # Write __init__.py files
        (src_dir / "__init__.py").write_text("", encoding="utf-8")
        (tests_dir / "__init__.py").write_text("", encoding="utf-8")

        # Step 2: Write relevant source files
        if relevant_files:
            for f in relevant_files:
                fd = f if isinstance(f, dict) else f.dict() if hasattr(f, 'dict') else {}
                file_path = fd.get("path", "")
                content = fd.get("content", "")
                if file_path and content:
                    target = workspace / file_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")
                    logger.debug("Wrote source file: %s", target)

        # Step 3: Apply the fix (write fix code)
        _apply_fix_to_workspace(fix_code, workspace, relevant_files)

        # Step 4: Write test file
        test_file = tests_dir / "test_fix.py"
        test_file.write_text(test_code, encoding="utf-8")
        logger.debug("Wrote test file: %s", test_file)

        # Step 5: Write requirements if any
        if requirements:
            req_file = workspace / "requirements.txt"
            req_file.write_text("\n".join(requirements), encoding="utf-8")

        # Step 6: Run Docker container
        result = _run_docker(str(workspace), requirements)

        return result

    except Exception as e:
        logger.error("Sandbox execution failed: %s", e)
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "tests_passed": 0,
            "tests_failed": 0,
        }

    finally:
        # Step 7: Cleanup temp directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.debug("Cleaned up temp dir: %s", temp_dir)
            except Exception as e:
                logger.warning("Failed to clean up temp dir: %s", e)


def _apply_fix_to_workspace(
    fix_code: str,
    workspace: Path,
    relevant_files: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Apply the fix code to files in the workspace.
    If fix_code looks like a diff, try to apply it.
    Otherwise, write it as a standalone file.
    """
    if not fix_code:
        return

    # Check if it's a unified diff
    if fix_code.startswith("---") or "+++ " in fix_code:
        # It's a diff — apply to the existing files
        _apply_unified_diff(fix_code, workspace)
    else:
        # It's raw code — write as a module file
        fix_file = workspace / "src" / "fix.py"
        fix_file.write_text(fix_code, encoding="utf-8")
        logger.debug("Wrote fix as standalone file: %s", fix_file)


def _apply_unified_diff(diff_text: str, workspace: Path) -> None:
    """
    Parse and apply a unified diff to files in the workspace.
    """
    current_file = None
    lines_to_remove = []
    lines_to_add = []

    for line in diff_text.split("\n"):
        if line.startswith("+++ b/") or line.startswith("+++ "):
            # Process previous file if any
            if current_file:
                _apply_changes_to_file(
                    workspace / current_file, lines_to_remove, lines_to_add
                )
            current_file = line.replace("+++ b/", "").replace("+++ ", "").strip()
            lines_to_remove = []
            lines_to_add = []
        elif line.startswith("--- "):
            continue
        elif line.startswith("@@"):
            continue
        elif current_file is not None:
            if line.startswith("+") and not line.startswith("+++"):
                lines_to_add.append(line[1:])
            elif line.startswith("-") and not line.startswith("---"):
                lines_to_remove.append(line[1:])

    # Process last file
    if current_file:
        _apply_changes_to_file(
            workspace / current_file, lines_to_remove, lines_to_add
        )


def _apply_changes_to_file(
    file_path: Path,
    removals: List[str],
    additions: List[str],
) -> None:
    """Apply line-level changes to a file."""
    if not file_path.exists():
        # New file — just write additions
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("\n".join(additions), encoding="utf-8")
        return

    content = file_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    # Remove specified lines
    for removal in removals:
        for i, line in enumerate(lines):
            if line.rstrip() == removal.rstrip():
                lines[i] = None
                break

    # Find insertion point and add new lines
    if removals:
        # Insert at the position of first removal
        insert_idx = None
        for i, line in enumerate(lines):
            if line is None:
                insert_idx = i
                break
        if insert_idx is not None:
            for j, addition in enumerate(additions):
                lines.insert(insert_idx + j, addition)

    # Remove None entries (removed lines)
    lines = [l for l in lines if l is not None]
    file_path.write_text("\n".join(lines), encoding="utf-8")


def _run_docker(
    workspace_path: str,
    requirements: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run pytest in a Docker container with security constraints.

    Args:
        workspace_path: Host path to mount into container.
        requirements:   Extra pip packages to install before running tests.

    Returns:
        dict: {success, output, error, tests_passed, tests_failed}
    """
    try:
        client = docker.from_env()
    except DockerException as e:
        logger.error("Failed to connect to Docker: %s", e)
        return {
            "success": False,
            "output": "",
            "error": f"Docker connection failed: {e}",
            "tests_passed": 0,
            "tests_failed": 0,
        }

    # Ensure the sandbox image exists
    try:
        client.images.get(DOCKER_IMAGE_NAME)
    except ImageNotFound:
        logger.warning(
            "Docker image '%s' not found. Building from Dockerfile...",
            DOCKER_IMAGE_NAME,
        )
        try:
            project_root = Path(__file__).parent.parent.parent
            client.images.build(
                path=str(project_root),
                dockerfile="docker/Dockerfile.sandbox",
                tag=DOCKER_IMAGE_NAME,
            )
            logger.info("Built Docker image: %s", DOCKER_IMAGE_NAME)
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Failed to build Docker image: {e}",
                "tests_passed": 0,
                "tests_failed": 0,
            }

    # Build the command
    cmd = "pytest /workspace/tests/ -v --tb=short"

    # Prepend pip install if extra requirements
    if requirements:
        pip_cmd = f"pip install {' '.join(requirements)} && "
        cmd = f"/bin/sh -c '{pip_cmd}{cmd}'"

    try:
        logger.info("Running Docker container with command: %s", cmd)

        container = client.containers.run(
            image=DOCKER_IMAGE_NAME,
            command=cmd,
            volumes={
                workspace_path: {
                    "bind": DOCKER_WORKSPACE_DIR,
                    "mode": "rw",
                }
            },
            network_mode="none",       # No internet access
            mem_limit=DOCKER_MEMORY_LIMIT,
            nano_cpus=int(float(DOCKER_CPU_LIMIT) * 1e9),
            remove=True,
            stdout=True,
            stderr=True,
            detach=False,
            timeout=DOCKER_TIMEOUT_SECONDS,
        )

        output = container.decode("utf-8") if isinstance(container, bytes) else str(container)
        result = _parse_pytest_output(output)
        result["output"] = output
        return result

    except ContainerError as e:
        # Container exited with non-zero — tests may have failed
        output = e.stderr.decode("utf-8") if e.stderr else ""
        stdout = e.container.logs(stdout=True, stderr=False).decode("utf-8") if hasattr(e, 'container') and e.container else ""

        full_output = stdout or output
        result = _parse_pytest_output(full_output)
        result["output"] = full_output
        result["error"] = output
        return result

    except Exception as e:
        logger.error("Docker run failed: %s", e)
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "tests_passed": 0,
            "tests_failed": 0,
        }


def _parse_pytest_output(output: str) -> Dict[str, Any]:
    """
    Parse pytest output to extract pass/fail counts.

    Example pytest summary line:
        '====== 3 passed, 1 failed in 0.12s ======'

    Returns:
        dict: {success, tests_passed, tests_failed, error}
    """
    tests_passed = 0
    tests_failed = 0
    success = False

    # Match pytest summary line
    passed_match = re.search(r"(\d+)\s+passed", output)
    failed_match = re.search(r"(\d+)\s+failed", output)
    error_match = re.search(r"(\d+)\s+error", output)

    if passed_match:
        tests_passed = int(passed_match.group(1))
    if failed_match:
        tests_failed = int(failed_match.group(1))

    errors = int(error_match.group(1)) if error_match else 0
    success = tests_failed == 0 and errors == 0 and tests_passed > 0

    return {
        "success": success,
        "output": "",
        "error": "" if success else f"{tests_failed} test(s) failed, {errors} error(s)",
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
    }
