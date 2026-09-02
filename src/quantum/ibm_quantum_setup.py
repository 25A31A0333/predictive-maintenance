"""
IBM Quantum & IBM Cloud Authentication and Setup Utility for Predictive Maintenance.

This module provides tools to:
1. Authenticate and save IBM Quantum API tokens (Qiskit Runtime).
2. Fetch access tokens from IBM Cloud IAM using an API key.
3. Query and select available quantum backends (e.g., ibm_brisbane, ibm_kyiv, simulator).
4. Configure PennyLane / Qiskit execution for the predictive maintenance quantum kernel.

Usage:
    # 1. Save and verify an IBM Quantum token:
    python src/quantum/ibm_quantum_setup.py --token <YOUR_IBM_QUANTUM_TOKEN>

    # 2. Check saved account & list quantum backends:
    python src/quantum/ibm_quantum_setup.py --list-backends

    # 3. Generate IAM Bearer token from IBM Cloud API Key:
    python src/quantum/ibm_quantum_setup.py --iam-api-key <YOUR_IBM_CLOUD_API_KEY>
"""

import os
import sys
import argparse
import json
import urllib.request
import urllib.parse
from typing import Optional, List, Dict, Any


def save_ibm_quantum_token(token: str, channel: str = "ibm_quantum_platform", overwrite: bool = True):
    """
    Saves and validates an IBM Quantum API token using Qiskit Runtime Service.
    """
    print(f"[*] Validating and saving IBM Quantum Token (channel: {channel})...")
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        QiskitRuntimeService.save_account(
            channel=channel,
            token=token,
            overwrite=overwrite
        )
        print("[SUCCESS] IBM Quantum token successfully saved to local Qiskit configuration!")
        
        # Test connection
        service = QiskitRuntimeService(channel=channel)
        backends = service.backends()
        print(f"[SUCCESS] Authenticated! Available IBM Quantum backends ({len(backends)} found):")
        for b in backends[:5]:
            status = "Operational" if b.status().operational else "Offline"
            print(f"  - {b.name} ({b.num_qubits} qubits, Status: {status})")
        if len(backends) > 5:
            print(f"  ... and {len(backends) - 5} more backends.")
    except ImportError:
        print("[WARN] 'qiskit-ibm-runtime' is not installed.")
        print("Saving token to .env file instead...")
        _save_token_to_env("IBM_QUANTUM_TOKEN", token)
        print("[SUCCESS] Saved token to .env.")
    except Exception as e:
        print(f"[ERROR] Failed to authenticate with IBM Quantum: {e}")
        _save_token_to_env("IBM_QUANTUM_TOKEN", token)


def list_ibm_backends(channel: str = "ibm_quantum_platform"):
    """Lists available IBM Quantum systems and simulators."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService(channel=channel)
        backends = service.backends()
        print(f"\n--- Available IBM Quantum Backends ({len(backends)}) ---")
        for b in backends:
            st = b.status()
            print(f"• Name: {b.name:<22} | Qubits: {b.num_qubits:<3} | Pending Jobs: {st.pending_jobs:<3} | Status: {'Active' if st.operational else 'Down'}")
    except ImportError:
        print("[ERROR] Please install qiskit-ibm-runtime: pip install qiskit-ibm-runtime")
    except Exception as e:
        print(f"[ERROR] Could not fetch backends: {e}")
        print("Ensure you have saved a valid IBM Quantum token using: python src/quantum/ibm_quantum_setup.py --token <YOUR_TOKEN>")


def get_ibm_cloud_iam_token(api_key: str) -> Optional[Dict[str, Any]]:
    """
    Exchanges an IBM Cloud API Key for an IAM OAuth Access Bearer Token.
    Endpoint: https://iam.cloud.ibm.com/identity/token
    """
    print("[*] Requesting IAM Access Token from IBM Cloud...")
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    data = urllib.parse.urlencode({
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key
    }).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            access_token = res_data.get("access_token")
            token_type = res_data.get("token_type", "Bearer")
            expires_in = res_data.get("expires_in", 3600)
            
            print("[SUCCESS] Successfully retrieved IBM Cloud IAM Access Token!")
            print(f"Token Type: {token_type}")
            print(f"Expires In: {expires_in} seconds (~{expires_in // 60} minutes)")
            print(f"Bearer Token Preview: {access_token[:20]}...{access_token[-10:]}")
            
            _save_token_to_env("IBM_CLOUD_IAM_TOKEN", access_token)
            return res_data
    except Exception as e:
        print(f"[ERROR] Failed to generate IAM token from API Key: {e}")
        return None


def _save_token_to_env(key_name: str, token_val: str):
    """Utility to write or update a token in .env."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    updated = False
    new_lines = []
    for line in lines:
        if line.startswith(f"{key_name}="):
            new_lines.append(f"{key_name}={token_val}\n")
            updated = True
        else:
            new_lines.append(line)
    if not updated:
        new_lines.append(f"{key_name}={token_val}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"[INFO] Saved {key_name} to {env_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IBM Quantum / IBM Cloud Token Setup for Predictive Maintenance")
    parser.add_argument("--token", type=str, help="IBM Quantum API Token (from quantum.ibm.com)")
    parser.add_argument("--channel", type=str, default="ibm_quantum_platform", choices=["ibm_quantum_platform", "ibm_cloud"], help="IBM Channel")
    parser.add_argument("--list-backends", action="store_true", help="List available IBM Quantum backends")
    parser.add_argument("--iam-api-key", type=str, help="IBM Cloud API Key to exchange for IAM Bearer token")

    args = parser.parse_args()

    if args.token:
        save_ibm_quantum_token(args.token, channel=args.channel)
    elif args.iam_api_key:
        get_ibm_cloud_iam_token(args.iam_api_key)
    elif args.list_backends:
        list_ibm_backends(channel=args.channel)
    else:
        parser.print_help()
        print("\n[TIP] Example:")
        print("  python src/quantum/ibm_quantum_setup.py --token <YOUR_IBM_QUANTUM_API_TOKEN>")
