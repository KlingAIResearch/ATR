#!/usr/bin/env python3
"""
Batch processing: Run all samples from test.jsonl with Qwen and Banana agents
Supports multi-GPU parallelization, with results in results/{agent_type}/{index}/ structure
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_single_case(
    case_index: int,
    test_case: dict,
    agent_type: str,
    worker_id: int,
    gpu_ids: List[int],
    verbose: bool = False
) -> Tuple[int, bool, str]:
    """
    Run a single test case
    
    Returns:
        (test_case_index, success, status_message)
    """
    test_case_index = test_case.get("index", case_index)
    image_path = test_case.get("input_image")
    instruction = test_case.get("instruction")
    
    if gpu_ids:
        gpu_id = gpu_ids[worker_id % len(gpu_ids)]
        cuda_env = str(gpu_id)
    else:
        cuda_env = ""
    
    temp_json_file = os.path.join(PROJECT_ROOT, f"temp_case_{test_case_index}_{worker_id}.json")
    try:
        with open(temp_json_file, 'w', encoding='utf-8') as f:
            json.dump(test_case, f, ensure_ascii=False, indent=2)
        
        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "run_edit.py"),
            "--json-file", temp_json_file,
            "--agent", agent_type
        ]
        
        env = os.environ.copy()
        if cuda_env:
            env['CUDA_VISIBLE_DEVICES'] = cuda_env
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600, env=env)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            msg = f"OK ({elapsed:.1f}s)"
            return (test_case_index, True, msg)
        else:
            error_msg = ""
            if result.stderr:
                error_lines = result.stderr.split('\n')
                error_msg = '\n'.join([line for line in error_lines if 'Error' in line or 'error' in line][-2:])
            if not error_msg and result.stdout:
                stdout_lines = result.stdout.split('\n')
                error_msg = '\n'.join([line for line in stdout_lines[-2:] if line.strip()])
            
            return (test_case_index, False, f"Failed: {error_msg[:80]}")
    
    except subprocess.TimeoutExpired:
        return (test_case_index, False, "Timeout (600s)")
    except Exception as e:
        return (test_case_index, False, str(e)[:80])
    finally:
        if os.path.exists(temp_json_file):
            try:
                os.remove(temp_json_file)
            except:
                pass

def run_jsonl_batch(jsonl_file: str, agent_type: str = "qwen", max_samples: int = None, verbose: bool = False):
    """
    Batch run test cases from JSONL file with specified agent
    """
    
    if not os.path.exists(jsonl_file):
        print(f"[✗] JSONL file not found: {jsonl_file}")
        return False
    
    test_cases = []
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if max_samples and idx >= max_samples:
                    break
                try:
                    case = json.loads(line.strip())
                    if case.get("input_image") and case.get("instruction"):
                        test_cases.append(case)
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"[✗] Error reading JSONL file: {e}")
        return False
        
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_env:
        gpu_ids = [int(g.strip()) for g in cuda_env.split(",") if g.strip()]
    else:
        gpu_ids = list(range(8))
        
    max_workers = len(gpu_ids) if gpu_ids else 1

    success_count = 0
    failed_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, test_case in enumerate(test_cases):
            future = executor.submit(
                run_single_case,
                case_index=i,
                test_case=test_case,
                agent_type=agent_type,
                worker_id=i,  
                gpu_ids=gpu_ids,
                verbose=verbose
            )
            futures.append(future)
            
        for future in as_completed(futures):
            try:
                idx, success, msg = future.result()
                if success:
                    print(f"  [✓] Case {idx} {msg}")
                    success_count += 1
                else:
                    print(f"  [✗] Case {idx} {msg}")
                    failed_count += 1
            except Exception as e:
                print(f"  [✗] Exception during parallel execution: {e}")
                failed_count += 1

    print(f"\nAgent: {agent_type.upper()} | Total: {len(test_cases)} | Success: {success_count} | Failed: {failed_count}")
    return failed_count == 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch run test cases with specified agents")
    parser.add_argument("--agent", choices=["qwen", "banana", "both"], default="both", help="Agent to use: qwen, banana, or both (default: both)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--max-samples", type=int, default=None, help="Max test cases to process")
    args = parser.parse_args()
    
    jsonl_file = os.path.join(PROJECT_ROOT, "test.jsonl")
    if not os.path.exists(jsonl_file):
        print(f"[✗] test.jsonl not found")
        return 1
    
    print(f"[*] Running batch tests from {jsonl_file}")
    print(f"[*] Using agent(s): {args.agent}")
    
    results = {}
    if args.agent in ["qwen", "both"]:
        results["qwen"] = run_jsonl_batch(jsonl_file, agent_type="qwen", max_samples=args.max_samples, verbose=args.verbose)
    
    if args.agent in ["banana", "both"]:
        results["banana"] = run_jsonl_batch(jsonl_file, agent_type="banana", max_samples=args.max_samples, verbose=args.verbose)
    
    print(f"\nResults:")
    for agent, success in results.items():
        print(f"  {agent.capitalize()}: {'OK' if success else 'FAILED'}")
    print(f"Output: {os.path.join(PROJECT_ROOT, 'results')}")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
