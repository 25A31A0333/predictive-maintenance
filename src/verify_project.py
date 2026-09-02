"""
Verification and Security Audit Script for Predictive Maintenance Project.
"""

import os
import glob
import py_compile
import re
import sys

def verify_all():
    print("=== 1. Checking Important Files ===")
    required_files = [
        "src/quantum/ibm_quantum_setup.py",
        "src/quantum/make_notebook.py",
        "src/quantum/run_in_the_quantum_computer.py",
        "push_to_github.py",
        ".gitignore",
        ".env.example",
        "Run_In_The_Quantum_Computer.ipynb",
        "notebooks/Run_In_The_Quantum_Computer.ipynb"
    ]
    for rf in required_files:
        exists = os.path.exists(rf)
        print(f"[{'OK' if exists else 'MISSING'}] {rf}")

    print("\n=== 2. Compiling & Syntax Checking all Python files ===")
    py_files = glob.glob("**/*.py", recursive=True)
    all_valid = True
    for pf in py_files:
        if ".venv" in pf or "__pycache__" in pf:
            continue
        try:
            py_compile.compile(pf, doraise=True)
            print(f"[VALID] {pf}")
        except Exception as e:
            print(f"[SYNTAX ERROR] {pf}: {e}")
            all_valid = False

    print("\n=== 3. Checking .gitignore Coverage ===")
    with open(".gitignore", "r", encoding="utf-8") as f:
        git_ignore_content = f.read()
    
    rules = [".env", "*.token", "*.key", "credentials.json", "__pycache__", ".pytest_cache"]
    for r in rules:
        if r in git_ignore_content:
            print(f"[COVERED] Rule '{r}' is active in .gitignore")
        else:
            print(f"[WARNING] Rule '{r}' missing from .gitignore")

    print("\n=== 4. Scanning for Hardcoded Secrets in Tracked Code ===")
    secret_patterns = [
        re.compile(r"ghp_[a-zA-Z0-9]{20,}"),
        re.compile(r"github_pat_[a-zA-Z0-9]{20,}"),
        re.compile(r"api[_-]?key\s*=\s*['\"][a-zA-Z0-9_-]{25,}['\"]")
    ]
    found_secret = False
    for ext in ["*.py", "*.ipynb", "*.md", "*.json"]:
        for fpath in glob.glob(f"**/{ext}", recursive=True):
            if ".git" in fpath or "__pycache__" in fpath:
                continue
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                for pat in secret_patterns:
                    if pat.search(content):
                        print(f"[SECURITY ALERT] Potential secret in {fpath}")
                        found_secret = True
    if not found_secret:
        print("[SAFE] Zero hardcoded secrets found across entire codebase.")

if __name__ == "__main__":
    verify_all()
