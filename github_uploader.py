# github_uploader.py
import os
import base64
import requests
import streamlit as st

def _get(name: str, default: str = "") -> str:
    # Prioridad: env vars (local) -> st.secrets (cloud)
    v = os.getenv(name)
    if v:
        return v
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

def upload_file_to_github_results(local_path: str, repo_path: str):
    """
    Sube/actualiza un archivo en el repo de RESULTADOS usando GitHub Contents API.
    Requiere:
      - GITHUB_TOKEN_RESULTS
      - GITHUB_OWNER_RESULTS
      - GITHUB_REPO_RESULTS
    Opcional:
      - GITHUB_BRANCH_RESULTS (default main)
    """
    token  = _get("GITHUB_TOKEN_RESULTS") or _get("GITHUB_TOKEN")
    owner  = _get("GITHUB_OWNER_RESULTS") or _get("GITHUB_USER")
    repo   = _get("GITHUB_REPO_RESULTS")  or _get("GITHUB_REPO")
    branch = _get("GITHUB_BRANCH_RESULTS") or _get("GITHUB_BRANCH") or "main"

    if not token or not owner or not repo:
        return False, "Faltan credenciales del repo de resultados (TOKEN/OWNER/REPO)."

    if not os.path.exists(local_path):
        return False, f"No existe el archivo local: {local_path}"

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{repo_path}"
    headers = {"Authorization": f"token {token}"}

    with open(local_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Buscar SHA si el archivo ya existe (para actualizar)
    sha = None
    r_get = requests.get(api_url, headers=headers, params={"ref": branch}, timeout=30)
    if r_get.status_code == 200:
        sha = r_get.json().get("sha")

    data = {
        "message": f"Subir resultados: {os.path.basename(local_path)}",
        "content": content_b64,
        "branch": branch,
    }
    if sha:
        data["sha"] = sha

    r_put = requests.put(api_url, headers=headers, json=data, timeout=30)
    if r_put.status_code in (200, 201):
        # html_url suele venir en el response (útil para mostrar link)
        try:
            html_url = r_put.json().get("content", {}).get("html_url", "")
        except Exception:
            html_url = ""
        return True, html_url or "OK"

    return False, f"GitHub error {r_put.status_code}: {r_put.text[:200]}"
