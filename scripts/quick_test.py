#!/usr/bin/env python3
"""
Quick test script - Run JSON test cases directly
"""

import os
import sys
import json
import subprocess
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

# Test case definition
TEST_CASE = {
    "index": 0,
    "input_image": "./examples/images/00000.jpg",
    "instruction": "Remove the 5 of spades card from the structure."
}

if not os.path.exists(TEST_CASE["input_image"]):
    print(f"[✗] Error: Image not found at {TEST_CASE['input_image']}")
    sys.exit(1)

json_str = json.dumps(TEST_CASE)

parser = argparse.ArgumentParser(description="Quick test script")
parser.add_argument("--agent", choices=["qwen", "banana", "both"], default="qwen", help="Agent to use: qwen, banana, or both (default: qwen)")
args = parser.parse_args()

agents = []
if args.agent in ["qwen", "both"]:
    agents.append("qwen")
if args.agent in ["banana", "both"]:
    agents.append("banana")

results = {}
for agent in agents:
    print(f"\n[*] Running test case with {agent.upper()} (index={TEST_CASE['index']})...")
    cmd = ["python", os.path.join("scripts", "run_edit.py"), "--test-json", json_str, "--agent", agent]
    exit_code = subprocess.call(cmd)
    results[agent] = exit_code == 0

print(f"\nResults:")
for agent, success in results.items():
    status = "[✓]" if success else "[✗]"
    print(f"  {status} {agent.capitalize()}: {'completed' if success else 'failed'}")

sys.exit(0 if all(results.values()) else 1)
