#!/usr/bin/env python3
# ==========================================
# ATR Framework - One-Click Editor
# ==========================================
# Usage: python run_edit.py --image <image_path> --instruction <instruction> [--output <output_dir>]
# Example: python run_edit.py --image input.jpg --instruction "Change the sky to sunset" --output ./results
# ==========================================

import os
import sys
import json
import argparse
import shutil
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# ==========================================
# 1. Setup Path & Import Core Modules
# ==========================================

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def _load_config_from_argv() -> None:
    if "--config" not in sys.argv:
        return
    try:
        config_path = sys.argv[sys.argv.index("--config") + 1]
    except IndexError:
        return
    try:
        from core.runtime_config import apply_runtime_config, load_runtime_config
        apply_runtime_config(load_runtime_config(config_path))
    except Exception as exc:
        print(f"[!] Warning: failed to load config before imports: {exc}")


_load_config_from_argv()

try:
    from core.captioner import generate_caption as _generate_caption
    from core.router import call_gemini, STAGE1_PROMPT_ABC, STAGE2_PROMPT_A1A2
    from core.agent_session_qwen import AgentSession as AgentSessionQwen
    from core.runtime_config import create_genai_client, get_qwen_image_edit_path
    from google import genai
except ImportError as e:
    print(f"[✗] Failed to import core modules: {e}")
    sys.exit(1)

# ==========================================
# 2. Configuration
# ==========================================

DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Google API Configuration
# Credentials are read from the GOOGLE_APPLICATION_CREDENTIALS environment variable.
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_PROJECT_ID")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("GOOGLE_LOCATION")
GEMINI_MODEL_NAME = "gemini-3-flash-preview"

# Editor Model Configuration
EDITOR_ID = get_qwen_image_edit_path("./examples/models/Qwen-Image-Edit-2509")

AGENT_TYPE = None
TEST_INDEX = None

# ==========================================
# 3. Core Pipeline Functions
# ==========================================

def validate_inputs(image_path: str, instruction: str) -> Tuple[bool, str]:
    """
    Validate input image and instruction.
    
    Returns:
        (is_valid, error_message)
    """
    # Check image exists
    if not os.path.exists(image_path):
        return False, f"Image not found: {image_path}"
    
    # Check image format
    valid_formats = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in valid_formats:
        return False, f"Unsupported image format: {ext}. Supported: {', '.join(valid_formats)}"
    
    # Check instruction
    if not instruction or len(instruction.strip()) == 0:
        return False, "Instruction cannot be empty"
    
    if len(instruction) > 500:
        return False, "Instruction too long (max 500 characters)"
    
    return True, ""


def create_output_directory(output_dir: str, agent_type: str = None, test_index: int = None) -> str:
    """
    Create output directory with structure: results/{agent_type}/{index}/
    If agent_type or test_index not provided, falls back to timestamped directory.
    
    Returns:
        Full path to output directory
    """
    if agent_type and test_index is not None:
        output_path = os.path.join(output_dir, agent_type, str(test_index))
    else:
        output_path = os.path.join(output_dir, f"edit_{TIMESTAMP}")
    
    os.makedirs(output_path, exist_ok=True)
    return output_path


def copy_input_image(image_path: str, output_dir: str) -> str:
    """
    Copy input image to output directory.
    
    Returns:
        Path to copied image
    """
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"input_{filename}")
    shutil.copy2(image_path, output_path)
    return output_path


def step_1_caption(image_path: str, instruction: str, output_dir: str, client=None) -> Dict:
    """
    Step 1: Generate image caption
    """
    print("\n" + "="*60)
    print("STEP 1: IMAGE UNDERSTANDING & CAPTIONING")
    print("="*60)
    
    try:
        if client is None:
            client = create_genai_client()
        
        caption_data = _generate_caption(image_path, instruction, client)
        
        if "error" in caption_data:
            raise Exception(caption_data["error"])
        
        caption_file = os.path.join(output_dir, "caption.json")
        with open(caption_file, "w", encoding="utf-8") as f:
            json.dump(caption_data, f, ensure_ascii=False, indent=2)
        
        return caption_data
    except Exception as e:
        print(f"[✗] Caption generation failed: {e}")
        raise


def step_2_router(instruction: str, caption_data: Dict, image_path: str, output_dir: str, client=None) -> Dict:
    """
    Step 2: Two-stage routing decision
    Stage 1: Classify into A/B/C
    Stage 2: If A, classify into A1/A2
    """
    print("\n" + "="*60)
    print("STEP 2: INTELLIGENT ROUTING & DECISION (Two-Stage)")
    print("="*60)
    
    try:
        if client is None:
            client = create_genai_client()
        
        # Stage 1: A/B/C Classification
        caption_str = json.dumps(caption_data)
        prompt_text_stage1 = STAGE1_PROMPT_ABC.format(
            caption=caption_str,
            instruction=instruction,
            edit_region_info="Edit region analysis included in caption"
        )
        
        routing_result_stage1 = call_gemini(prompt_text_stage1, image_path, client)
        
        if "error" in routing_result_stage1:
            stage1_class = "A1"
            routing_data = {"class": stage1_class, "reasoning": "Default routing due to error"}
            return routing_data
        
        stage1_class = routing_result_stage1.get("class", "A")
        stage1_reasoning = routing_result_stage1.get("reasoning", "")
        
        # Stage 2: A1/A2 Classification (if Stage 1 is A)
        if stage1_class == "A":
            prompt_text_stage2 = STAGE2_PROMPT_A1A2.format(
                caption=caption_str,
                instruction=instruction,
                edit_region_info="Edit region analysis included in caption"
            )
            
            routing_result_stage2 = call_gemini(prompt_text_stage2, image_path, client)
            
            if "error" in routing_result_stage2:
                final_class = "A1"
                final_reasoning = "Stage 2 failed, defaulting to A1"
            else:
                final_class = routing_result_stage2.get("class", "A1")
                final_reasoning = routing_result_stage2.get("reasoning", "")
            
            routing_data = {
                "stage1": stage1_class,
                "stage1_reasoning": stage1_reasoning,
                "class": final_class,
                "reasoning": final_reasoning
            }
        else:
            routing_data = {
                "stage1": stage1_class,
                "class": stage1_class,
                "reasoning": stage1_reasoning
            }
        
        routing_file = os.path.join(output_dir, "routing.json")
        with open(routing_file, "w", encoding="utf-8") as f:
            json.dump(routing_data, f, ensure_ascii=False, indent=2)
        
        return routing_data
    except Exception as e:
        print(f"[✗] Routing failed: {e}")
        return {"class": "A1", "reasoning": f"Error in routing: {str(e)}"}


def step_3_execute(
    instruction: str,
    image_path: str,
    routing_class: str,
    output_dir: str,
    test_index=0,
    agent_type: str = "qwen",
    editor_pipe=None
) -> Dict:
    """
    Step 3: Execute editing based on routing decision
    """
    print("\n" + "="*60)
    print(f"STEP 3: EXECUTION - PIPELINE {routing_class} [{agent_type.upper()}]")
    print("="*60)
    
    try:
        if agent_type.lower() == "banana":
            from core.agent_session_banana import AgentSession as AgentSessionBanana
            session = AgentSessionBanana(
                index=test_index,
                original_instruction=instruction,
                working_instruction=instruction,
                original_path=image_path,
                output_dir=output_dir,
                routing_class=routing_class
            )
        else:
            session = AgentSessionQwen(
                index=test_index,
                original_instruction=instruction,
                working_instruction=instruction,
                original_path=image_path,
                output_dir=output_dir,
                editor_pipe=editor_pipe,
                routing_class=routing_class
            )
        
        final_image_path = session.run_loop()
        
        trace_path = os.path.join(output_dir, "trace.json")
        if os.path.exists(trace_path):
            with open(trace_path, "r", encoding="utf-8") as f:
                trace_data = json.load(f)
        else:
            trace_data = {"status": "error", "message": "No trace file generated"}
        
        return trace_data
    except Exception as e:
        print(f"[✗] Execution failed: {e}")
        raise


def find_output_image(output_dir: str) -> Optional[str]:
    """
    Find the final edited image in output directory.
    
    Returns:
        Path to output image or None
    """
    # Look for image files
    image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
    
    for pattern in image_patterns:
        from glob import glob
        files = glob(os.path.join(output_dir, pattern))
        # Exclude input image
        output_files = [f for f in files if "input_" not in f]
        if output_files:
            # Return the most recently modified
            return max(output_files, key=os.path.getmtime)
    
    return None


def generate_report(
    instruction: str,
    caption_data: Dict,
    routing_data: Dict,
    trace_data: Dict,
    output_dir: str
) -> str:
    """
    Generate a comprehensive report
    """
    report = {
        "timestamp": TIMESTAMP,
        "instruction": instruction,
        "pipeline": {
            "caption": caption_data,
            "routing": routing_data,
            "execution": trace_data
        }
    }
    
    report_file = os.path.join(output_dir, "report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    return report_file


# ==========================================
# 4. Main Pipeline
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="🌟 ATR Framework - One-Click Image Editor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_edit.py --image photo.jpg --instruction "Change the sky to blue"
  
  # With custom output directory
  python run_edit.py \\
    --image photo.jpg \\
    --instruction "Remove the person from background" \\
    --output ./my_results
  
  # Using JSON test case
  python run_edit.py --test-json '{"index": 0, "input_image": "...", "instruction": "..."}'
  
  # Get help
  python run_edit.py --help
        """
    )
    
    parser.add_argument("--image", required=False, help="Path to input image")
    parser.add_argument("--instruction", required=False, help="Editing instruction")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--test-json", required=False, help="JSON test case string")
    parser.add_argument("--json-file", required=False, help="Path to JSON file")
    parser.add_argument("--agent", required=False, default="qwen", help="Agent: qwen or banana")
    parser.add_argument("--config", required=False, help="Path to runtime config JSON")
    
    args = parser.parse_args()
    
    image_path = None
    instruction = None
    test_index = 0
    
    if args.json_file:
        try:
            with open(args.json_file, 'r', encoding='utf-8') as f:
                test_case = json.load(f)
            image_path = test_case.get("input_image")
            instruction = test_case.get("instruction")
            test_index = test_case.get("index", 0)
            
            if not image_path or not instruction:
                print("[✗] JSON file must contain 'input_image' and 'instruction' fields")
                return 1
                
        except Exception as e:
            print(f"[✗] Failed to read JSON file: {e}")
            return 1
            
    elif args.test_json:
        try:
            test_case = json.loads(args.test_json)
            image_path = test_case.get("input_image")
            instruction = test_case.get("instruction")
            test_index = test_case.get("index", 0)
            
            if not image_path or not instruction:
                print("[✗] JSON must contain 'input_image' and 'instruction' fields")
                return 1
            
        except json.JSONDecodeError as e:
            print(f"[✗] Failed to parse JSON: {e}")
            return 1
            
    else:
        if not args.image or not args.instruction:
            print("[✗] Please provide --image and --instruction, OR --json-file, OR --test-json")
            parser.print_help()
            return 1
        image_path = args.image
        instruction = args.instruction
        test_index = 0
    
    print("\n" + "="*60)
    print("ATR FRAMEWORK - ONE-CLICK EDITOR")
    print("="*60)
    
    is_valid, error_msg = validate_inputs(image_path, instruction)
    if not is_valid:
        print(f"[✗] Input validation failed: {error_msg}")
        return 1
    
    try:
        output_base_dir = args.output if args.output else DEFAULT_OUTPUT_DIR
        output_dir = create_output_directory(output_base_dir, agent_type=args.agent, test_index=test_index)
    except Exception as e:
        print(f"[✗] Failed to create output directory: {e}")
        return 1
    
    try:
        input_image = copy_input_image(image_path, output_dir)
    except Exception as e:
        print(f"[✗] Failed to copy input image: {e}")
        return 1
    
    print("[*] Initializing Gemini client...")
    try:
        gemini_client = create_genai_client()
    except Exception as e:
        print(f"[!] Warning: Could not initialize Gemini client: {e}")
        gemini_client = None
    
    editor_pipe = None
    if args.agent.lower() == "qwen":
        print("[*] Initializing Qwen-Edit model...")
        try:
            if torch.cuda.is_available():
                device = f"cuda:0"
            else:
                device = "cpu"
            
            editor_pipe = QwenImageEditPlusPipeline.from_pretrained(
                EDITOR_ID, torch_dtype=torch.bfloat16
            ).to(device)
            editor_pipe.set_progress_bar_config(disable=True)
        except Exception as e:
            print(f"[!] Warning: Could not initialize Qwen-Edit model: {e}")
            return 1
    try:
        caption_data = step_1_caption(image_path, instruction, output_dir, gemini_client)
        routing_data = step_2_router(instruction, caption_data, image_path, output_dir, gemini_client)
        trace_data = step_3_execute(
            instruction=instruction,
            image_path=image_path,
            routing_class=routing_data.get("class", "A1"),
            output_dir=output_dir,
            test_index=test_index,
            agent_type=args.agent,
            editor_pipe=editor_pipe
        )
        
        generate_report(instruction, caption_data, routing_data, trace_data, output_dir)
        output_image = find_output_image(output_dir)
        
        print("\n" + "="*60)
        print("SUCCESS! Pipeline completed")
        print("="*60)
        print(f"Results saved to: {output_dir}")
        if output_image:
            print(f"Output image: {output_image}")
        print("="*60 + "\n")
        
        return 0
        
    except Exception as e:
        print("\n" + "="*60)
        print("FAILED! Pipeline error")
        print("="*60)
        print(f"Error: {e}")
        print("="*60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
