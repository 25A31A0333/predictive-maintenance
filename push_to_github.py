"""
Helper script to push the repository to GitHub.

Usage:
    python push_to_github.py --remote-url https://github.com/25A31A0333/predictive-maintenance.git
    python push_to_github.py --token <YOUR_GITHUB_PERSONAL_ACCESS_TOKEN> --repo-name predictive-maintenance
"""

import argparse
import sys
from dulwich import porcelain
from dulwich.repo import Repo


def push_repo(remote_url: str, branch: str = "main"):
    repo = Repo(".")
    print(f"[*] Pushing local branch '{branch}' to remote: {remote_url}")
    try:
        porcelain.push(repo, remote_url, refspecs=[f"refs/heads/{branch}:refs/heads/{branch}".encode("ascii")])
        print(f"[SUCCESS] Successfully pushed to GitHub ({remote_url})!")
    except Exception as e:
        print(f"[ERROR] Failed to push: {e}")
        print("\n[NOTE] If authentication is required, use:")
        print("    https://<YOUR_GITHUB_TOKEN>@github.com/25A31A0333/<REPO_NAME>.git")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push to GitHub")
    parser.add_argument(
        "--remote-url",
        type=str,
        default="https://github.com/25A31A0333/predictive-maintenance.git",
        help="GitHub remote URL",
    )
    parser.add_argument("--branch", type=str, default="master", help="Branch name")
    args = parser.parse_args()

    push_repo(args.remote_url, args.branch)
