"""
src/tools/github_tools.py — GitHub API wrapper functions using PyGithub.

All tools use the GITHUB_TOKEN environment variable for auth.
Every function includes error handling with exponential backoff on 429 rate limits.

Functions:
    get_issue()          — Fetch issue title, body, labels, comments
    search_code()        — Search repo codebase for keywords
    get_file_content()   — Read file content from a specific branch
    list_directory()     — List files/folders at a repo path
    create_branch()      — Create a new branch from an existing one
    commit_file()        — Commit a file to a branch
    open_pull_request()  — Open a PR from head branch to base branch
"""

import logging
import os
import time
from typing import Dict, List, Optional

from github import Auth, Github, GithubException, RateLimitExceededException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_github_client() -> Github:
    """
    Create an authenticated PyGithub client.

    Reads:  GITHUB_TOKEN from environment.
    Raises: RuntimeError if GITHUB_TOKEN is not set.
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN is not set. "
            "Create a token at https://github.com/settings/tokens "
            "with scopes: repo (read+write) and pull_requests (write)."
        )
    auth = Auth.Token(token)
    return Github(auth=auth)


def _retry_on_rate_limit(func, *args, max_retries: int = 3, **kwargs):
    """
    Wrapper that retries a function call with exponential backoff on 429 rate limits.

    Args:
        func:        Callable to execute.
        max_retries: Max retry attempts (default 3).
    """
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except RateLimitExceededException:
            if attempt == max_retries:
                logger.error("GitHub rate limit exceeded after %d retries.", max_retries)
                raise
            wait = 2 ** attempt * 10  # 10s, 20s, 40s
            logger.warning(
                "GitHub rate limit hit. Retrying in %ds (attempt %d/%d).",
                wait, attempt + 1, max_retries,
            )
            time.sleep(wait)
        except GithubException as e:
            logger.error("GitHub API error: %s", e)
            raise


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------


def get_issue(repo_owner: str, repo_name: str, issue_number: int) -> Dict:
    """
    Fetch a GitHub issue's details.

    Reads:  Nothing from state — called by Orchestrator or Research Agent.
    Writes: Returns dict consumed by orchestrator to populate state fields.

    Args:
        repo_owner:   GitHub username or org, e.g. 'octocat'
        repo_name:    Repository name, e.g. 'hello-world'
        issue_number: Issue number, e.g. 42

    Returns:
        dict with keys: title, body, labels, comments
    """
    def _fetch():
        g = _get_github_client()
        repo = g.get_repo(f"{repo_owner}/{repo_name}")
        issue = repo.get_issue(number=issue_number)
        comments = [c.body for c in issue.get_comments()]
        labels = [l.name for l in issue.labels]
        return {
            "title": issue.title,
            "body": issue.body or "",
            "labels": labels,
            "comments": comments,
        }

    logger.info("Fetching issue #%d from %s/%s", issue_number, repo_owner, repo_name)
    return _retry_on_rate_limit(_fetch)


def search_code(repo_full_name: str, query: str) -> List[Dict]:
    """
    Search for code in a GitHub repository matching a query string.

    Reads:  Nothing from state — called by Research Agent.
    Writes: Returns list of matching file paths.

    Args:
        repo_full_name: 'owner/repo' format, e.g. 'octocat/hello-world'
        query:          Search query string

    Returns:
        list of dicts: [{path, url, score}]
    """
    def _search():
        g = _get_github_client()
        results = g.search_code(query=f"{query} repo:{repo_full_name}")
        matches = []
        for item in results[:20]:  # Limit to 20 results
            matches.append({
                "path": item.path,
                "url": item.html_url,
                "score": getattr(item, "score", 0),
            })
        return matches

    logger.info("Searching code in %s for: '%s'", repo_full_name, query)
    return _retry_on_rate_limit(_search)


def get_file_content(
    repo_full_name: str, file_path: str, branch: str = "main"
) -> str:
    """
    Read the raw content of a file from a GitHub repository.

    Reads:  Nothing from state — called by Research Agent.
    Writes: Returns raw file content string.

    Args:
        repo_full_name: 'owner/repo' format
        file_path:      Path within the repo, e.g. 'src/math_utils.py'
        branch:         Branch to read from (default 'main')

    Returns:
        Raw file content as a string.
    """
    def _fetch():
        g = _get_github_client()
        repo = g.get_repo(repo_full_name)
        content_file = repo.get_contents(file_path, ref=branch)
        if isinstance(content_file, list):
            raise ValueError(f"'{file_path}' is a directory, not a file.")
        return content_file.decoded_content.decode("utf-8")

    logger.info("Reading file %s from %s (branch: %s)", file_path, repo_full_name, branch)
    return _retry_on_rate_limit(_fetch)


def list_directory(repo_full_name: str, path: str = "") -> List[str]:
    """
    List files and subdirectories at a given path in a GitHub repository.

    Reads:  Nothing from state — called by Research Agent.
    Writes: Returns list of filenames.

    Args:
        repo_full_name: 'owner/repo' format
        path:           Directory path within the repo (empty string for root)

    Returns:
        list of file/folder names at the given path.
    """
    def _fetch():
        g = _get_github_client()
        repo = g.get_repo(repo_full_name)
        contents = repo.get_contents(path)
        if not isinstance(contents, list):
            contents = [contents]
        return [
            f"{item.name}/" if item.type == "dir" else item.name
            for item in contents
        ]

    logger.info("Listing directory '%s' in %s", path or "/", repo_full_name)
    return _retry_on_rate_limit(_fetch)


def create_branch(
    repo_full_name: str,
    branch_name: str,
    from_branch: str = "main",
) -> bool:
    """
    Create a new branch in a GitHub repository.

    Reads:  Nothing from state — called by PR Writer Agent.
    Writes: Creates branch on GitHub; returns True on success.

    Args:
        repo_full_name: 'owner/repo' format
        branch_name:    New branch name, e.g. 'fix/issue-42'
        from_branch:    Source branch to branch from (default 'main')

    Returns:
        True if the branch was created successfully.
    """
    def _create():
        g = _get_github_client()
        repo = g.get_repo(repo_full_name)
        source = repo.get_branch(from_branch)
        try:
            repo.create_git_ref(
                ref=f"refs/heads/{branch_name}",
                sha=source.commit.sha,
            )
        except GithubException as e:
            # Branch may already exist — that's OK
            if e.status == 422:
                logger.warning("Branch '%s' already exists, continuing.", branch_name)
                return True
            raise
        return True

    logger.info("Creating branch '%s' from '%s' in %s", branch_name, from_branch, repo_full_name)
    return _retry_on_rate_limit(_create)


def commit_file(
    repo_full_name: str,
    branch: str,
    file_path: str,
    new_content: str,
    commit_message: str,
) -> bool:
    """
    Commit (create or update) a file on a branch in a GitHub repository.

    Reads:  Nothing from state — called by PR Writer Agent.
    Writes: Commits file to GitHub; returns True on success.

    Args:
        repo_full_name: 'owner/repo' format
        branch:         Target branch
        file_path:      Path for the file in the repo
        new_content:    New file content
        commit_message: Commit message

    Returns:
        True if commit succeeded.
    """
    def _commit():
        g = _get_github_client()
        repo = g.get_repo(repo_full_name)
        try:
            # Try to get existing file to update
            existing = repo.get_contents(file_path, ref=branch)
            if isinstance(existing, list):
                raise ValueError(f"'{file_path}' is a directory, cannot commit.")
            repo.update_file(
                path=file_path,
                message=commit_message,
                content=new_content,
                sha=existing.sha,
                branch=branch,
            )
            logger.info("Updated file '%s' on branch '%s'", file_path, branch)
        except GithubException as e:
            if e.status == 404:
                # File doesn't exist — create it
                repo.create_file(
                    path=file_path,
                    message=commit_message,
                    content=new_content,
                    branch=branch,
                )
                logger.info("Created file '%s' on branch '%s'", file_path, branch)
            else:
                raise
        return True

    logger.info("Committing to %s: %s on branch '%s'", repo_full_name, file_path, branch)
    return _retry_on_rate_limit(_commit)


def open_pull_request(
    repo_full_name: str,
    title: str,
    body: str,
    head_branch: str,
    base_branch: str = "main",
) -> str:
    """
    Open a pull request on a GitHub repository.

    Reads:  Nothing from state — called by PR Writer Agent.
    Writes: Creates PR on GitHub; returns PR URL string.

    Args:
        repo_full_name: 'owner/repo' format
        title:          PR title
        body:           PR body (Markdown)
        head_branch:    Branch with the changes
        base_branch:    Target branch to merge into (default 'main')

    Returns:
        The full URL of the opened pull request.
    """
    def _open():
        g = _get_github_client()
        repo = g.get_repo(repo_full_name)
        pr = repo.create_pull(
            title=title,
            body=body,
            head=head_branch,
            base=base_branch,
        )
        logger.info("Opened pull request: %s", pr.html_url)
        return pr.html_url

    logger.info(
        "Opening PR in %s: '%s' (%s → %s)",
        repo_full_name, title, head_branch, base_branch,
    )
    return _retry_on_rate_limit(_open)
