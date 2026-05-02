import os

import requests


def create_github_issue(title: str, body: str, logger) -> None:
    repo = os.getenv("GH_REPO")
    token = os.getenv("GH_TOKEN")
    if not repo or not token:
        logger.warning("GH_REPO/GH_TOKEN not set; skipping GitHub issue creation.")
        return

    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    payload = {"title": title, "body": body}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
    except requests.RequestException as e:
        logger.warning(f"Failed to create issue: {type(e).__name__}: {e}")
        return

    if response.status_code >= 300:
        logger.warning(
            f"Failed to create issue: {response.status_code} {response.text[:200]}"
        )
    else:
        logger.info("GitHub issue created.")
