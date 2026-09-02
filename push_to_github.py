"""
Automated GitHub Push Utility for Predictive Maintenance.
Pushes code directly to GitHub repository https://github.com/25A31A0333/predictive-maintenance

Usage:
    # 1. Push with Personal Access Token:
    python push_to_github.py --token <YOUR_GITHUB_TOKEN>

    # 2. Interactive push (prompts for token securely):
    python push_to_github.py

    # 3. Custom repository or branch:
    python push_to_github.py --token <YOUR_TOKEN> --user 25A31A0333 --repo predictive-maintenance --branch main
"""

import os
import sys
import getpass
import argparse
from typing import Optional


def ensure_repo_committed():
    """Stages all modified and new files and commits them if needed."""
    try:
        from dulwich import porcelain
        from dulwich.repo import Repo
        repo = Repo(".")
        status = porcelain.status(repo)
        
        has_changes = bool(status.staged["add"] or status.staged["delete"] or status.staged["modify"] or status.unstaged or status.untracked)
        
        # Filter out .git, temp caches, or pycache
        untracked = [p for p in status.untracked if not p.startswith(b".pytest_cache") and not p.startswith(b"__pycache__")]
        
        if status.unstaged or untracked:
            print("[*] Staging untracked & modified files...")
            porcelain.add(repo, paths=None)
            porcelain.commit(
                repo,
                message=b"Update Predictive Maintenance repository with latest quantum models & dashboard",
                author=b"25A31A0333 <thotakiran67@gmail.com>"
            )
            print("[SUCCESS] Committed latest changes locally.")
            
        # Ensure main and master branch refs point to current HEAD
        head = repo.head()
        repo.refs[b"refs/heads/main"] = head
        repo.refs[b"refs/heads/master"] = head
    except Exception as e:
        print(f"[WARN] Local commit check notice: {e}")


def push_to_github(
    token: Optional[str] = None,
    username: str = "25A31A0333",
    repo_name: str = "predictive-maintenance",
    branch: str = "main",
    remote_url: Optional[str] = None
):
    """Pushes the current repository to GitHub using Dulwich."""
    ensure_repo_committed()

    from dulwich import porcelain
    from dulwich.repo import Repo
    repo = Repo(".")

    if not token and not remote_url:
        print("\n=======================================================")
        print(" GitHub Authentication for Account: " + username)
        print("=======================================================")
        print("To push to GitHub, a Personal Access Token (PAT) is required.")
        print("Generate one here: https://github.com/settings/tokens (Scope: 'repo')\n")
        try:
            token = input("Enter GitHub Personal Access Token (or paste here): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[ABORTED] Push cancelled.")
            sys.exit(1)

    if token:
        # Construct authenticated URL
        target_url = f"https://{username}:{token}@github.com/{username}/{repo_name}.git"
    elif remote_url:
        target_url = remote_url
    else:
        target_url = f"https://github.com/{username}/{repo_name}.git"

    print(f"\n[*] Uploading to https://github.com/{username}/{repo_name} (branch: {branch})...")
    try:
        # Push main branch
        ref_spec = f"refs/heads/{branch}:refs/heads/{branch}".encode("ascii")
        porcelain.push(repo, target_url, refspecs=[ref_spec])
        print(f"\n[SUCCESS] 🎉 Successfully uploaded project to https://github.com/{username}/{repo_name} on branch '{branch}'!")
        
        # Also sync master branch if pushing main
        if branch == "main":
            try:
                master_spec = b"refs/heads/master:refs/heads/master"
                porcelain.push(repo, target_url, refspecs=[master_spec])
                print(f"[SUCCESS] Synced 'master' branch as well.")
            except Exception:
                pass
                
    except Exception as e:
        print(f"\n[ERROR] Push failed: {e}")
        print("\n[TROUBLESHOOTING]:")
        print("1. Verify your Personal Access Token has the 'repo' scope checked.")
        print("   Create a new classic token at: https://github.com/settings/tokens/new")
        print(f"2. Ensure the repository https://github.com/{username}/{repo_name} exists.")
        print("3. Run with token:")
        print(f"   python push_to_github.py --token <YOUR_GITHUB_TOKEN>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push Predictive Maintenance project to GitHub")
    parser.add_argument("--token", type=str, help="GitHub Personal Access Token (PAT)")
    parser.add_argument("--user", type=str, default="25A31A0333", help="GitHub username")
    parser.add_argument("--repo", type=str, default="predictive-maintenance", help="Repository name")
    parser.add_argument("--branch", type=str, default="main", help="Branch name (default: main)")
    parser.add_argument("--remote-url", type=str, help="Direct remote URL")

    args = parser.parse_args()

    push_to_github(
        token=args.token,
        username=args.user,
        repo_name=args.repo,
        branch=args.branch,
        remote_url=args.remote_url
    )
