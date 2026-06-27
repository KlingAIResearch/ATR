#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import traceback
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_config import apply_runtime_config, load_runtime_config


def read_jsonl(jsonl_file: str, max_samples: Optional[int] = None) -> list[dict[str, Any]]:
    cases = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not item.get("input_image") or not item.get("instruction"):
                raise ValueError(f"JSONL item missing input_image or instruction at line {idx + 1}")
            cases.append(item)
    return cases


def normalize_gpu_ids(value: Any) -> list[int]:
    if value is None:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible:
            return [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
        return []
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    raise TypeError("gpu_ids must be a list, comma-separated string, or null")


def load_qwen_pipe(run_edit_module):
    torch = run_edit_module.torch
    pipeline_cls = run_edit_module.QwenImageEditPlusPipeline

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Worker loading Qwen-Image-Edit on {device}: {run_edit_module.EDITOR_ID}", flush=True)
    pipe = pipeline_cls.from_pretrained(
        run_edit_module.EDITOR_ID,
        torch_dtype=torch.bfloat16,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def read_existing_trace_status(output_dir: str, agent: str, case_index: str) -> Optional[str]:
    sample_output_dir = os.path.join(output_dir, agent, case_index)
    trace_path = os.path.join(sample_output_dir, "trace.json")
    if not os.path.exists(trace_path):
        return None

    try:
        with open(trace_path, "r", encoding="utf-8") as f:
            trace_data = json.load(f)
        return trace_data.get("status")
    except Exception:
        return None


def should_skip_case(
    case: dict[str, Any],
    output_dir: str,
    agent: str,
    skip_statuses: set[str],
) -> tuple[bool, Optional[str]]:
    if not skip_statuses:
        return False, None

    case_index = str(case.get("index", "0"))
    status = read_existing_trace_status(output_dir, agent, case_index)
    return status in skip_statuses, status


def run_case_in_worker(
    run_edit_module,
    case: dict[str, Any],
    output_dir: str,
    agent: str,
    gemini_client,
    editor_pipe,
) -> tuple[str, bool, str]:
    case_index = str(case.get("index", "0"))
    image_path = case.get("input_image")
    instruction = case.get("instruction")

    is_valid, error_msg = run_edit_module.validate_inputs(image_path, instruction)
    if not is_valid:
        return case_index, False, f"input validation failed: {error_msg}"

    sample_output_dir = run_edit_module.create_output_directory(
        output_dir,
        agent_type=agent,
        test_index=case_index,
    )

    try:
        run_edit_module.copy_input_image(image_path, sample_output_dir)
        caption_data = run_edit_module.step_1_caption(
            image_path,
            instruction,
            sample_output_dir,
            gemini_client,
        )
        routing_data = run_edit_module.step_2_router(
            instruction,
            caption_data,
            image_path,
            sample_output_dir,
            gemini_client,
        )
        trace_data = run_edit_module.step_3_execute(
            instruction=instruction,
            image_path=image_path,
            routing_class=routing_data.get("class", "A1"),
            output_dir=sample_output_dir,
            test_index=case_index,
            agent_type=agent,
            editor_pipe=editor_pipe,
        )
        run_edit_module.generate_report(
            instruction,
            caption_data,
            routing_data,
            trace_data,
            sample_output_dir,
        )
        return case_index, True, sample_output_dir
    except Exception as exc:
        traceback.print_exc()
        return case_index, False, f"{type(exc).__name__}: {exc}"


def worker_main(
    worker_id: int,
    gpu_id: Optional[int],
    config_path: str,
    output_dir: str,
    agent: str,
    task_queue,
    result_queue,
) -> None:
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    config = load_runtime_config(config_path)
    apply_runtime_config(config)

    scripts_dir = PROJECT_ROOT / "scripts"
    sys.path.insert(0, str(scripts_dir))
    import run_edit as run_edit_module
    from core.runtime_config import create_genai_client

    print(
        f"[*] Worker {worker_id} started. gpu={gpu_id if gpu_id is not None else 'inherit'} agent={agent}",
        flush=True,
    )

    gemini_client = create_genai_client()
    editor_pipe = load_qwen_pipe(run_edit_module) if agent == "qwen" else None

    while True:
        case = task_queue.get()
        if case is None:
            break
        case_index, ok, message = run_case_in_worker(
            run_edit_module,
            case,
            output_dir,
            agent,
            gemini_client,
            editor_pipe,
        )
        result_queue.put((agent, worker_id, case_index, ok, message))

    print(f"[*] Worker {worker_id} finished.", flush=True)


def run_agent(
    config_path: str,
    config: dict[str, Any],
    cases: list[dict[str, Any]],
    agent: str,
) -> bool:
    output_dir = config["output_dir"]
    gpu_ids = normalize_gpu_ids(config.get("gpu_ids"))
    max_workers = int(config.get("max_workers") or (len(gpu_ids) if gpu_ids else 1))
    if gpu_ids:
        max_workers = min(max_workers, len(gpu_ids))
    skip_statuses = set(str(s) for s in config.get("skip_statuses", []))

    pending_cases = []
    skipped_count = 0
    skipped_by_status: dict[str, int] = {}
    for case in cases:
        skip, status = should_skip_case(case, output_dir, agent, skip_statuses)
        if skip:
            skipped_count += 1
            skipped_by_status[status or "unknown"] = skipped_by_status.get(status or "unknown", 0) + 1
        else:
            pending_cases.append(case)

    print(
        f"[*] Agent {agent}: total={len(cases)}, skipped={skipped_count}, "
        f"pending={len(pending_cases)}, skip_statuses={sorted(skip_statuses)}",
        flush=True,
    )
    if skipped_by_status:
        print(f"[*] Skipped by status: {skipped_by_status}", flush=True)

    if not pending_cases:
        print(f"[*] Agent {agent}: nothing to run.", flush=True)
        return True

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    for case in pending_cases:
        task_queue.put(case)
    for _ in range(max_workers):
        task_queue.put(None)

    processes = []
    for worker_id in range(max_workers):
        gpu_id = gpu_ids[worker_id] if gpu_ids else None
        proc = ctx.Process(
            target=worker_main,
            args=(worker_id, gpu_id, config_path, output_dir, agent, task_queue, result_queue),
        )
        proc.start()
        processes.append(proc)

    success_count = 0
    failed_count = 0
    total = len(pending_cases)
    received = 0

    while received < total:
        try:
            agent_name, worker_id, case_index, ok, message = result_queue.get(timeout=30)
        except queue.Empty:
            alive = any(proc.is_alive() for proc in processes)
            if not alive:
                break
            continue

        received += 1
        if ok:
            success_count += 1
            print(f"  [OK] [{agent_name}/worker{worker_id}] Case {case_index}: {message}", flush=True)
        else:
            failed_count += 1
            print(f"  [FAILED] [{agent_name}/worker{worker_id}] Case {case_index}: {message}", flush=True)

    for proc in processes:
        proc.join()

    for proc in processes:
        if proc.exitcode != 0:
            failed_count += 1
            print(f"  [FAILED] worker process exitcode={proc.exitcode}", flush=True)

    print(
        f"Agent {agent}: total={total}, received={received}, "
        f"success={success_count}, failed={failed_count}",
        flush=True,
    )
    return failed_count == 0 and received == total


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ImgEdit pipeline with resident model workers from one config JSON."
    )
    parser.add_argument("--config", required=True, help="Path to imgedit pipeline config JSON.")
    args = parser.parse_args()

    config_path = str(Path(args.config).resolve())
    config = load_runtime_config(config_path)
    apply_runtime_config(config)

    if not config.get("jsonl_file"):
        raise ValueError("config.jsonl_file is required")
    if not config.get("output_dir"):
        raise ValueError("config.output_dir is required")

    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
    cases = read_jsonl(config["jsonl_file"], config.get("max_samples"))

    agent = config.get("agent", "qwen")
    if agent not in {"qwen", "banana", "both"}:
        raise ValueError("config.agent must be qwen, banana, or both")
    agents = ["qwen", "banana"] if agent == "both" else [agent]

    print(f"[*] Config: {config_path}")
    print(f"[*] JSONL: {config['jsonl_file']}")
    print(f"[*] Output: {config['output_dir']}")
    print(f"[*] Cases: {len(cases)}")
    print(f"[*] Agent(s): {', '.join(agents)}")
    print("[*] Resident mode: each worker loads model once and reuses it.")

    all_ok = True
    for agent_name in agents:
        print(f"\n===== Running agent: {agent_name} =====", flush=True)
        all_ok = run_agent(config_path, config, cases, agent_name) and all_ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
