import base64
import os
from typing import Dict, Optional, Tuple

import requests
import streamlit as st

GITHUB_API = "https://api.github.com"


def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read from st.secrets first, then env vars."""
    try:
        if key in st.secrets:
            val = st.secrets.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    except Exception:
        pass

    env_val = os.getenv(key)
    if isinstance(env_val, str) and env_val.strip():
        return env_val.strip()

    return default


def _get_results_config() -> Dict[str, str]:
    """
    Supports either:
      - top-level secrets:
          GITHUB_TOKEN_RESULTS, GITHUB_OWNER_RESULTS, GITHUB_REPO_RESULTS, GITHUB_BRANCH_RESULTS
      - or a table:
          [github_results]
          token = "..."
          owner = "..."
          repo  = "..."
          branch = "main"
    """
    cfg: Dict[str, str] = {}

    # Table-style secrets
    try:
        table = st.secrets.get("github_results", {})
        if isinstance(table, dict):
            cfg["token"] = (table.get("token") or "").strip()
            cfg["owner"] = (table.get("owner") or "").strip()
            cfg["repo"] = (table.get("repo") or "").strip()
            cfg["branch"] = (table.get("branch") or "").strip()
    except Exception:
        pass

    # Fallback to top-level keys
    cfg["token"] = cfg.get("token") or (_get_secret("GITHUB_TOKEN_RESULTS") or "")
    cfg["owner"] = cfg.get("owner") or (_get_secret("GITHUB_OWNER_RESULTS") or "")
    cfg["repo"] = cfg.get("repo") or (_get_secret("GITHUB_REPO_RESULTS") or "")
    cfg["branch"] = cfg.get("branch") or (_get_secret("GITHUB_BRANCH_RESULTS", "main") or "main")

    return cfg


def _headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def upload_bytes_to_github_results(
    content_bytes: bytes,
    repo_path: str,
    commit_message: Optional[str] = None,
    branch: Optional[str] = None,
) -> Tuple[bool, Dict[str, str]]:
    """
    Upload content to GitHub using the Contents API.

    Returns: (ok, info)
      - ok=True -> info contains at least "html_url" when available
      - ok=False -> info contains "error"
    """
    cfg = _get_results_config()
    token = cfg.get("token", "")
    owner = cfg.get("owner", "")
    repo = cfg.get("repo", "")
    branch = (branch or cfg.get("branch") or "main").strip()

    if not token or not owner or not repo:
        return False, {
            "error": "Faltan secretos: GITHUB_TOKEN_RESULTS / GITHUB_OWNER_RESULTS / GITHUB_REPO_RESULTS (y opcional GITHUB_BRANCH_RESULTS)."  # noqa: E501
        }

    # 1) Check if file exists to get sha (needed to update)
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{repo_path}"
    params = {"ref": branch} if branch else None
    r_get = requests.get(url, headers=_headers(token), params=params, timeout=30)

    sha = None
    if r_get.status_code == 200:
        try:
            sha = r_get.json().get("sha")
        except Exception:
            sha = None
    elif r_get.status_code != 404:
        return False, {"error": f"GET falló ({r_get.status_code}): {r_get.text}"}

    # 2) PUT content
    content_b64 = base64.b64encode(content_bytes).decode("utf-8")
    payload: Dict[str, str] = {
        "message": commit_message or f"Add {repo_path}",
        "content": content_b64,
    }
    if branch:
        payload["branch"] = branch
    if sha:
        payload["sha"] = sha

    r_put = requests.put(url, headers=_headers(token), json=payload, timeout=30)
    if r_put.status_code not in (200, 201):
        return False, {"error": f"PUT falló ({r_put.status_code}): {r_put.text}"}

    data = r_put.json()
    # GitHub returns: {content: {html_url, ...}, commit: {...}}
    html_url = (data.get("content") or {}).get("html_url") or ""
    return True, {"html_url": html_url}


def upload_text_to_github_results(
    text: str,
    repo_path: str,
    commit_message: Optional[str] = None,
    branch: Optional[str] = None,
    encoding: str = "utf-8",
) -> Tuple[bool, Dict[str, str]]:
    return upload_bytes_to_github_results(
        text.encode(encoding),
        repo_path=repo_path,
        commit_message=commit_message,
        branch=branch,
    )


def upload_file_to_github_results(
    local_path: str,
    repo_path: str,
    commit_message: Optional[str] = None,
    branch: Optional[str] = None,
) -> Tuple[bool, Dict[str, str]]:
    """Backward-compatible: uploads a local file."""
    with open(local_path, "rb") as f:
        content = f.read()
    return upload_bytes_to_github_results(
        content,
        repo_path=repo_path,
        commit_message=commit_message,
        branch=branch,
    )
