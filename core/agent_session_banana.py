# ==========================================
# ReAct Agent: Dynamic Vision-Based Editing
# ==========================================
import os
import re
import io
import json
import time
import shutil
import sys
import traceback
from typing import Any, Dict, Optional, Tuple, List

import torch
from PIL import Image, ImageDraw
import torch.multiprocessing as mp

# ==========================================
# 0. System Path Setup & Imports
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ATR_ROOT = os.path.dirname(SCRIPT_DIR)
TOOL_DIR = os.path.join(ATR_ROOT, "tools")

if ATR_ROOT not in sys.path:
    sys.path.insert(0, ATR_ROOT)
if TOOL_DIR not in sys.path:
    sys.path.append(TOOL_DIR)

from google import genai
from google.genai import types
from core.runtime_config import create_genai_client, get_gemini_model

# ==========================================
# 1. Config & Auth
# ==========================================
# Credentials are read from the GOOGLE_APPLICATION_CREDENTIALS environment variable.
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_PROJECT_ID")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("GOOGLE_LOCATION")
GEMINI_MODEL_NAME = get_gemini_model("gemini-3-flash-preview")

META_JSON = "./examples/data/data_decisions_filtered.jsonl"
OUT_ROOT  = "./examples/output/banana_pipeline"
TRACE_FILENAME = "trace.json"

# ==========================================
# 2. Tool Loading
# ==========================================
try:
    from tools.crop_tool import crop_tool
    from tools.fixprompt_tool import fixprompt_tool
    from tools.croppaste_tool import paste_with_seamless_clone as croppaste_tool
    from tools.sam3_tool import sam3_tool
    from tools.smartpaste_tool import smartpaste_tool
    from tools.reprompt_tool import reprompt_tool
    from tools.ifinish_tool import ifinish_tool
except ImportError as e:
    print(f"[FATAL] Failed to import tools: {e}")
    print(f"  Tool directory: {TOOL_DIR}")
    print(f"  Available: {os.listdir(TOOL_DIR) if os.path.exists(TOOL_DIR) else 'NOT FOUND'}")
    sys.exit(1)


# ==========================================
# 3. Per-Class System Prompts
# ==========================================
SYSTEM_PROMPT_DIR = os.path.join(ATR_ROOT, "prompts_banana")

def _load_class_prompt(filepath):
    """Load a system prompt from a txt file.
    Supports both plain text files and files containing a Python variable
    assignment in the format:  VAR = r\"\"\"...content...\"\"\".strip()
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        m = re.search(r'r"""(.*?)"""\s*\.strip\(\)', content, re.S)
        if m:
            return m.group(1).strip()
        return content.strip()
    except FileNotFoundError:
        print(f"[Warn] System prompt file not found: {filepath}, falling back to default.")
        return ""

_CLASS_PROMPTS = {
    "A1": _load_class_prompt(os.path.join(SYSTEM_PROMPT_DIR, "system_A1.txt")),
    "A2": _load_class_prompt(os.path.join(SYSTEM_PROMPT_DIR, "system_A2.txt")),
    "B":  _load_class_prompt(os.path.join(SYSTEM_PROMPT_DIR, "system_B.txt")),
    "C":  _load_class_prompt(os.path.join(SYSTEM_PROMPT_DIR, "system_C.txt")),
}

# ==========================================
# 4. Agent Session Class
# ==========================================
class AgentSession:
    def __init__(self, index, original_instruction, working_instruction, original_path, output_dir, routing_class=None):
        self.index = str(index)
        self.original_instruction = original_instruction
        self.instruction = working_instruction
        self.original_path = original_path
        self.output_dir = output_dir
        self.routing_class = routing_class
        
        self.current_image_path = original_path
        self.history = []
        self.step_count = 0
        self.fixprompt_used = False
        self.max_steps = 10
        
        self.file_map = {"original": original_path}
        
        self.step_bbox_cache = {}
        self.step_sam3_cache = {}
        
        self.trace_path = os.path.join(output_dir, TRACE_FILENAME)
        self.trace_data = {
            "index": self.index,
            "meta": {"instruction": original_instruction, "input_image": original_path, "routing_class": routing_class},
            "steps": [],
            "status": "running"
        }
        self._save_trace()

    def _save_trace(self):
        with open(self.trace_path, "w", encoding="utf-8") as f:
            json.dump(self.trace_data, f, ensure_ascii=False, indent=2)

    def _draw_bboxes(self, img_path, orig_bbox, target_bbox, out_path):
        try:
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            W, H = img.size
            def to_abs(rel): return [rel[0]/1000*W, rel[1]/1000*H, rel[2]/1000*W, rel[3]/1000*H]
            
            if orig_bbox: 
                draw.rectangle(to_abs(orig_bbox), outline="red", width=5)
            if target_bbox: 
                draw.rectangle(to_abs(target_bbox), outline="blue", width=5)
                
            img.save(out_path)
        except Exception as e:
            print(f"  [Warn] Failed to draw debug bbox: {e}")
            shutil.copy(img_path, out_path)

    def _adjust_bbox_to_gemini_ratio(self, image_path, bbox_rel1000):
        """
        Adjust bbox_rel1000 based on Gemini supported aspect ratios.
        Follows expand-only principle and ensures minimum crop region of 400x300 pixels.
        """
        try:
            with Image.open(image_path) as img:
                W_img, H_img = img.size
        except Exception as e:
            print(f"  [Warn] Failed to read image for bbox adjustment: {e}")
            return bbox_rel1000

        x0_rel, y0_rel, x1_rel, y1_rel = bbox_rel1000

        x0 = x0_rel * W_img / 1000.0
        y0 = y0_rel * H_img / 1000.0
        x1 = x1_rel * W_img / 1000.0
        y1 = y1_rel * H_img / 1000.0

        w = x1 - x0
        h = y1 - y0

        if w <= 0 or h <= 0:
            return bbox_rel1000

        MIN_W = 400
        MIN_H = 300
        
        base_w = max(w, MIN_W)
        base_h = max(h, MIN_H)
        
        current_ratio = base_w / base_h

        # Gemini 2.5 Flash Image supported aspect ratios
        supported_ratios = {
            "1:1": 1024 / 1024,
            "2:3": 832 / 1248,
            "3:2": 1248 / 832,
            "3:4": 864 / 1184,
            "4:3": 1184 / 864,
            "4:5": 896 / 1152,
            "5:4": 1152 / 896,
            "9:16": 768 / 1344,
            "16:9": 1344 / 768,
            "21:9": 1536 / 672
        }
        
        target_ratio_name = min(supported_ratios, key=lambda k: abs(supported_ratios[k] - current_ratio))
        target_ratio = supported_ratios[target_ratio_name]

        new_w, new_h = base_w, base_h
        if current_ratio < target_ratio:
            new_w = base_h * target_ratio
        else:
            new_h = base_w / target_ratio

        pad_x = new_w - w
        pad_y = new_h - h

        new_x0 = x0 - pad_x / 2.0
        new_x1 = x1 + pad_x / 2.0
        new_y0 = y0 - pad_y / 2.0
        new_y1 = y1 + pad_y / 2.0

        # Boundary protection: shift if out of bounds
        if new_x0 < 0:
            new_x1 += (0 - new_x0)
            new_x0 = 0
        if new_x1 > W_img:
            new_x0 -= (new_x1 - W_img)
            new_x1 = W_img

        if new_y0 < 0:
            new_y1 += (0 - new_y0)
            new_y0 = 0
        if new_y1 > H_img:
            new_y0 -= (new_y1 - H_img)
            new_y1 = H_img

        new_x0 = max(0, min(new_x0, W_img))
        new_x1 = max(0, min(new_x1, W_img))
        new_y0 = max(0, min(new_y0, H_img))
        new_y1 = max(0, min(new_y1, H_img))

        out_x0_rel = int(new_x0 / W_img * 1000)
        out_y0_rel = int(new_y0 / H_img * 1000)
        out_x1_rel = int(new_x1 / W_img * 1000)
        out_y1_rel = int(new_y1 / H_img * 1000)

        return [out_x0_rel, out_y0_rel, out_x1_rel, out_y1_rel]

    def _build_prompt(self):
        history_text = ""
        if not self.history:
            history_text = "No steps executed yet. Start with Step 1."
        else:
            for item in self.history:
                s_idx = item['step']
                tool = item['tool']
                params = json.dumps(item['params'])
                out_name = item['output_name']
                history_text += f"Step {s_idx} [{tool}]: Params={params} -> Output: {out_name}\n"

        if self.step_count == 0:
            fixprompt_hint = ""
            if self.routing_class in ("B", "C") and not self.fixprompt_used:
                fixprompt_hint = (
                    "Additional available tool:\n"
                    "If the instruction is vague, underspecified, uses unclear references or pronouns, "
                    "contains spatial ambiguity, or requires reasoning about the target object, attributes, relationships, "
                    "or intended edit before it becomes directly executable, "
                    "you may call `fixprompt_tool` to rewrite it before formal planning starts.\n"
                    "This tool does NOT edit the image. If you use it, the system will replace the instruction "
                    "with the returned fixed instruction and restart formal planning from Step 1 on the same original image.\n"
                    "If the instruction is already clear, do NOT call this tool.\n"
                    "Required format:\n"
                    "<step>\n"
                    "tool: fixprompt_tool\n"
                    "source: original\n"
                    "instruction: the instruction to refine\n"
                    "</step>"
                )
            user_msg = (
                f"{fixprompt_hint}\n\n"
                f"**INSTRUCTION**: {self.instruction}\n\n"
                f"**CURRENT IMAGE**: This is the Original Image.\n\n"
                f"**TASK**: Analyze the image and formulate a strategy. "
                f"You MUST respond with exactly one `<think>...</think>` block for your analysis, "
                f"immediately followed by exactly ONE `<step>...</step>` block for your first action."
            )
        else:
            current_image_desc = f"Result of Step {self.step_count}."
            user_msg = (
                f"**INSTRUCTION**: {self.instruction}\n\n"
                f"**CURRENT IMAGE**: {current_image_desc}\n"
                f"**HISTORY**:\n{history_text}\n\n"
                f"**TASK**: Verify the previous step success. "
                f"You MUST respond with exactly one `<think>...</think>` block for your reasoning, "
                f"immediately followed by exactly ONE `<step>...</step>` block for your next action. "
            )

        return user_msg

    def _can_use_fixprompt_step0(self):
        return self.routing_class in ("B", "C") and self.step_count == 0 and not self.fixprompt_used

    def _execute_fixprompt_step0(self, step_data):
        src_name = step_data.get("source", "original")
        src = self.file_map.get(src_name, self.file_map.get("original"))
        raw_instruction = step_data.get("instruction", self.instruction)
        fixed_instruction = fixprompt_tool(src, raw_instruction)

        self.instruction = fixed_instruction
        self.fixprompt_used = True
        step_data["source"] = src_name
        step_data["raw_instruction"] = raw_instruction
        step_data["fixed_instruction"] = fixed_instruction
        return fixed_instruction

    def _call_gemini_planner(self, prompt_text, image_path=None):
        client = create_genai_client()
        
        # Use the explicitly provided image (e.g. debug visualization) if given,
        # otherwise fall back to the current working image.
        img_path_to_read = image_path if image_path and os.path.exists(image_path) else self.current_image_path
        if not os.path.exists(img_path_to_read):
            raise FileNotFoundError(f"Missing current image: {img_path_to_read}")
            
        with open(img_path_to_read, 'rb') as f:
            img_bytes = f.read()

        system_prompt = _CLASS_PROMPTS.get(self.routing_class, "")

        final_prompt = prompt_text + "\n\nStart your response directly with: <think>"
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=[
                    system_prompt,
                    types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'),
                    final_prompt
                ],
                config=types.GenerateContentConfig(max_output_tokens=8192)
            )
            return response.text
        except Exception as e:
            print(f"   [Planner Error] {e}")
            return None

    def _parse_step(self, text):
        if not text: return None, None, False
        
        if "<done/>" in text or "<done>" in text:
            return text, None, True
            
        step_match = re.search(r"<step>(.*?)</step>", text, flags=re.S)
        
        thought = "No thought provided."
        
        think_match = re.search(r"<think>(.*?)</think>", text, flags=re.S)
        
        if think_match:
            thought = think_match.group(1).strip()
        elif step_match:
            pre_step_text = text[:step_match.start()].strip()
            if pre_step_text:
                thought = pre_step_text
        else:
            thought = text.strip()

        if not step_match:
            print(f"   [Parse Warning] No <step> tag found. Raw output:\n{text[:200]}...")
            return thought, None, False
            
        body = step_match.group(1).strip()
        step_data = {}
        for line in body.splitlines():
            if ":" not in line: continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if k in ["bbox_rel1000", "offset_rel1000", "crop_keywords", "crop_keyword"]:
                try: 
                    v = json.loads(v)
                except: 
                    pass
            step_data[k] = v
            
        return thought, step_data, False


    def _execute_tool(self, step_data, step_idx):
        tool = step_data.get("tool")
        final_name = f"process{step_idx}"
        final_path = os.path.join(self.output_dir, f"{final_name}.png")
        step_data["output"] = final_name
        
        print(f"  [Exec] {tool} -> {final_name}")
        
        def get_path(name):
            return self.file_map.get(name, self.file_map.get("original"))

        debug_vis_path = None

        try:
            if tool == "sam3_tool":
                src = get_path(step_data.get("source"))
                keyword = step_data.get("crop_keywords") or step_data.get("crop_keyword")
                sam_out_dir = os.path.join(self.output_dir, "sam3_outputs")
                res = sam3_tool(src, keyword, output_dir=sam_out_dir)
                
                is_valid_mask = False
                if res["status"] == "success":
                    info = res["best_mask_info"]
                    bbox = info.get("bbox_rel1000", [0,0,0,0])
                    if (bbox[2] - bbox[0] > 0) and (bbox[3] - bbox[1] > 0):
                        is_valid_mask = True
                
                if is_valid_mask:
                    info = res["best_mask_info"]
                    self.step_bbox_cache[final_name] = info["bbox_rel1000"]
                    self.step_sam3_cache[final_name] = info["cutout_path"]
                    shutil.copy(res["visualization_path"], final_path)
                    debug_vis_path = final_path
                else:
                    print(f"  [SAM3 Fail] Empty mask for keyword: {keyword}")
                    shutil.copy(src, final_path)
                    # Soft failure: let the loop continue so the model can see the error
                    # in history and retry sam3_tool with a different crop_keyword.
                    step_data["ERROR"] = f"sam3_tool failed to find '{keyword}'. Mask is EMPTY. Retry with different crop_keywords (Remember to use a LIST)."

            elif tool == "target_tool":
                src = get_path(step_data.get("source"))
                offset = step_data.get("offset_rel1000")
                src_step_name = step_data.get("source_bbox_step")
                
                orig_bbox = self.step_bbox_cache.get(src_step_name)
                
                if offset and orig_bbox:
                    dx, dy = offset
                    w = orig_bbox[2] - orig_bbox[0]
                    h = orig_bbox[3] - orig_bbox[1]
                    
                    raw_tx0 = orig_bbox[0] + dx
                    raw_ty0 = orig_bbox[1] + dy
                    
                    tx0 = max(0, min(1000 - w, raw_tx0))
                    ty0 = max(0, min(1000 - h, raw_ty0))
                    
                    tx1 = tx0 + w
                    ty1 = ty0 + h
                    
                    target_bbox = [tx0, ty0, tx1, ty1]
                    self.step_bbox_cache[final_name] = target_bbox
                    
                    debug_img_name = f"{final_name}_debug.png"
                    debug_vis_path = os.path.join(self.output_dir, debug_img_name)
                    
                    shutil.copy(src, final_path)
                    self._draw_bboxes(src, orig_bbox, target_bbox, debug_vis_path)
                else:
                    print("  [Target Fail] Missing cache or offset")
                    shutil.copy(src, final_path)
                    step_data["ERROR"] = f"target_tool failed: bbox cache for '{src_step_name}' not found. You must run sam3_tool successfully before calling target_tool."

            elif tool == "smartpaste_tool":
                src = get_path(step_data.get("source"))
                crop_step_name = step_data.get("crop_step")
                target_step_name = step_data.get("target_step")
                
                cutout_path = self.step_sam3_cache.get(crop_step_name)
                target_bbox = self.step_bbox_cache.get(target_step_name)
                
                if cutout_path and target_bbox:
                    res = smartpaste_tool(src, cutout_path, target_bbox, self.output_dir, self.index, str(step_idx))
                    if res["status"] == "success":
                        shutil.copy(res["output_path"], final_path)
                    else:
                        shutil.copy(src, final_path)
                        step_data["ERROR"] = f"smartpaste_tool execution failed internally: {res.get('error', 'unknown error')}. The object was not pasted."
                else:
                    print("  [Paste Fail] Missing cache")
                    shutil.copy(src, final_path)
                    step_data["ERROR"] = f"smartpaste_tool failed: cutout or target bbox not found (crop_step='{crop_step_name}', target_step='{target_step_name}'). Ensure sam3_tool and target_tool completed successfully first."

            elif tool == "crop_tool":
                src = get_path(step_data.get("source"))
                last_tool = self.history[-1]["tool"] if self.history else None
                if last_tool == "crop_tool":
                    print(f"  [Rule Violation] Consecutive crop_tool prevented.")
                    shutil.copy(src, final_path)
                    step_data["ERROR"] = "CRITICAL RULE VIOLATION: You cannot use crop_tool twice in a row! You MUST use 'qwen_edit' right now to edit the crop you just made, and then use 'croppaste_tool' to paste it back."
                    
                else:
                    raw_bbox = step_data.get("bbox_rel1000")
                    if not raw_bbox: raise ValueError("crop_tool missing bbox")
                    
                    # Intercept and adjust bbox for Gemini compatibility
                    bbox = self._adjust_bbox_to_gemini_ratio(src, raw_bbox)
                    print(f"  [Crop Adjust] Bbox {raw_bbox} -> {bbox}")
                    
                    ret = crop_tool(src, f"{self.index}_s{step_idx}", bbox, self.output_dir)
                    if os.path.exists(ret["output_path"]):
                        shutil.move(ret["output_path"], final_path)
                    else:
                        raise FileNotFoundError("Crop tool output missing")

            elif tool == "fixprompt_tool":
                raise ValueError("fixprompt_tool is only available as step0 before formal planning starts.")

            elif tool == "qwen_edit":
                src = get_path(step_data.get("source", "original"))
                prompt = step_data.get("prompt", self.instruction)
                image = Image.open(src).convert("RGB")
                orig_w, orig_h = image.size 

                current_ratio = orig_w / orig_h
                supported_ratios_map = {
                    "1:1": 1024/1024, "2:3": 832/1248, "3:2": 1248/832, 
                    "3:4": 864/1184, "4:3": 1184/864, "4:5": 896/1152, 
                    "5:4": 1152/896, "9:16": 768/1344, "16:9": 1344/768, "21:9": 1536/672
                }
                best_ratio_str = min(supported_ratios_map, key=lambda k: abs(supported_ratios_map[k] - current_ratio))

                edit_client = create_genai_client()
                response = edit_client.models.generate_content(
                    model="gemini-2.5-flash-image",
                    contents=[prompt, image],
                    config=types.GenerateContentConfig(
                        image_config=types.ImageConfig(
                            aspect_ratio=best_ratio_str,
                        )
                    )
                )

                saved = False
                for part in response.parts:
                    if part.text is not None:
                        print(f"  [GeminiEdit] {part.text}")
                    elif part.inline_data is not None:
                        img_bytes = part.inline_data.data
                        real_pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        resized_image = real_pil_image.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
                        resized_image.save(final_path)
                        saved = True
                if not saved:
                    raise RuntimeError("gemini-2.5-flash-image returned no image output.")

            elif tool == "croppaste_tool":
                bg_name = step_data.get("source")
                fg_name = step_data.get("crop_source")
                raw_bbox = step_data.get("bbox_rel1000")
                
                bg_path = get_path(bg_name)
                fg_path = get_path(fg_name)
                
                if not raw_bbox: raise ValueError("croppaste missing bbox")
                bbox = self._adjust_bbox_to_gemini_ratio(bg_path, raw_bbox)
                print(f"  [Paste Adjust] Bbox {raw_bbox} -> {bbox}")

                croppaste_tool(bg_path, fg_path, bbox, final_path)

            elif tool == "refine_edit_tool":
                src = get_path(step_data.get("source"))
                instr = step_data.get("instruction", "Fix artifacts")
                try:
                    reprompt_res = reprompt_tool(self.original_path, src, instr)
                    gen_prompt = reprompt_res["instruction"]
                except:
                    gen_prompt = instr
                
                image = Image.open(src).convert("RGB")
                edit_client = create_genai_client()
                response = edit_client.models.generate_content(
                    model="gemini-2.5-flash-image",
                    contents=[gen_prompt, image],
                )

                saved = False
                for part in response.parts:
                    if part.text is not None:
                        print(f"  [GeminiRefine] {part.text}")
                    elif part.inline_data is not None:
                        part.as_image().save(final_path)
                        saved = True

                if not saved:
                    raise RuntimeError("gemini-2.5-flash-image returned no image output.")

            elif tool == "ifinish_tool":
                orig_src = get_path(step_data.get("source", "original"))
                curr_src = get_path(step_data.get("current_step"))
                
                print(f"  [Verify] Calling ifinish_tool on {os.path.basename(curr_src)}")
                
                finish_res = ifinish_tool(
                    orig_src, 
                    curr_src, 
                    self.original_instruction, 
                    PROJECT_ID, 
                    LOCATION, 
                    GEMINI_MODEL_NAME
                )
                
                step_data["ifinish_result"] = finish_res
                shutil.copy(curr_src, final_path)

            else:
                print(f"  [Warn] Tool {tool} not implemented, copying source.")
                src = get_path(step_data.get("source"))
                shutil.copy(src, final_path)

            self.file_map[final_name] = final_path
            return final_path, "success", debug_vis_path

        except Exception as e:
            print(f"  [Exec Except] {e}")
            traceback.print_exc()
            if not os.path.exists(final_path):
                try: shutil.copy(get_path("original"), final_path)
                except: pass
            return final_path, "failed", None


    def run_loop(self):
        print(f"[{self.index}] Starting ReAct Loop. Goal: {self.instruction}")
        
        img_for_planner = self.current_image_path
        
        while self.step_count < self.max_steps:
            if self.step_count == self.max_steps - 1:
                print(f"[{self.index}] Step 10 reached — forcing qwen_edit with original instruction.")
                step_data = {
                    "tool": "qwen_edit",
                    "source": "original",
                    "prompt": self.original_instruction,
                    "output": "process10"
                }
                out_path, status, _ = self._execute_tool(step_data, self.max_steps)
                step_record = {
                    "step": self.max_steps,
                    "thought": "Forced fallback: qwen_edit with original instruction at step 10.",
                    "tool": "qwen_edit",
                    "params": step_data,
                    "status": status,
                    "output_name": "process10"
                }
                self.trace_data["steps"].append(step_record)
                self.trace_data["status"] = "completed_forced"
                if status == "success":
                    self.current_image_path = out_path
                self._save_trace()
                break

            prompt = self._build_prompt()
            
            resp_text = self._call_gemini_planner(prompt, image_path=img_for_planner)
            if not resp_text:
                print(f"[{self.index}] Planner API failed.")
                self.trace_data["status"] = "api_error"
                self._save_trace()
                break
            
            print(f"\n[{self.index}] --- Step {self.step_count + 1} Raw Output ---")
            print(resp_text)
            print("-" * 50)
            
            thought, step_data, is_done = self._parse_step(resp_text)
            
            step_record = {
                "step": self.step_count + 1,
                "input_context": {
                    "seen_image": img_for_planner,
                    "read_prompt": prompt
                },
                "thought": thought, 
                "raw_response": resp_text
            }
            
            if is_done:
                print(f"[{self.index}] Agent signalled DONE.")
                self.trace_data["status"] = "completed"
                step_record["action"] = "done"
                self.trace_data["steps"].append(step_record)
                self._save_trace()
                break
            
            if not step_data:
                print(f"[{self.index}] Failed to parse step.")
                self.trace_data["status"] = "parse_error"
                step_record["action"] = "parse_error"
                self.trace_data["steps"].append(step_record)
                self._save_trace()
                break

            if step_data.get("tool") == "fixprompt_tool":
                step_record["step"] = 0
                step_record["tool"] = "fixprompt_tool"
                step_record["params"] = step_data
                step_record["output_name"] = "instruction_only"

                if not self._can_use_fixprompt_step0():
                    step_record["status"] = "failed"
                    step_record["error"] = "fixprompt_tool is only available once before formal Step 1 for routing class B/C."
                    self.trace_data["steps"].append(step_record)
                    self.trace_data["status"] = "execution_failed"
                    self._save_trace()
                    break

                try:
                    fixed_instruction = self._execute_fixprompt_step0(step_data)
                    step_record["status"] = "success"
                    step_record["raw_instruction"] = step_data.get("raw_instruction", "")
                    step_record["fixed_instruction"] = fixed_instruction
                    self.trace_data["steps"].append(step_record)
                    self._save_trace()
                    img_for_planner = self.current_image_path
                    print(f"[{self.index}] fixprompt_tool completed as step0. Restarting formal Step 1.")
                    continue
                except Exception as e:
                    step_record["status"] = "failed"
                    step_record["error"] = str(e)
                    step_record["exception_type"] = type(e).__name__
                    self.trace_data["steps"].append(step_record)
                    self.trace_data["status"] = "execution_failed"
                    self._save_trace()
                    break
                 
            out_path, status, debug_vis_path = self._execute_tool(step_data, self.step_count + 1)
            
            step_record["tool"] = step_data.get("tool")
            step_record["params"] = step_data
            step_record["status"] = status
            step_record["output_name"] = f"process{self.step_count + 1}"
            
            self.trace_data["steps"].append(step_record)
            self._save_trace()
            
            if status == "success":
                self.current_image_path = out_path
                self.history.append(step_record)
                self.step_count += 1

                if step_data.get("tool") == "ifinish_tool":
                    ifinish_res = step_data.get("ifinish_result")
                    finished = (
                        ifinish_res is True
                        or ifinish_res == "true"
                        or (isinstance(ifinish_res, dict) and ifinish_res.get("finished") in (True, "true"))
                        or (isinstance(ifinish_res, dict) and ifinish_res.get("result") in (True, "true"))
                        or (isinstance(ifinish_res, dict) and ifinish_res.get("is_finished") in (True, "true"))
                    )
                    if finished:
                        print(f"[{self.index}] ifinish_tool returned True. Editing complete.")
                        self.trace_data["status"] = "completed"
                        self._save_trace()
                        break
                
                if debug_vis_path and os.path.exists(debug_vis_path):
                    img_for_planner = debug_vis_path
                else:
                    img_for_planner = out_path
            
            else:
                print(f"[{self.index}] Execution failed.")
                self.trace_data["status"] = "execution_failed"
                self._save_trace()
                break
        
        else:
            print(f"[{self.index}] Max steps ({self.max_steps}) reached.")
            self.trace_data["status"] = "max_steps_reached"
            self._save_trace()

        return self.current_image_path

# ==========================================
# 5. Worker & Main
# ==========================================
def gpu_worker(rank, world_size, data_list):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    print(f"[GPU {rank}] Init Qwen-Edit...")
    

    my_chunk = data_list[rank::world_size]
    
    for item in my_chunk:
        idx = str(item["index"])
        instr = item["instruction"]
        img_path = item["input_image"]
        routing_class = item.get("class", "A1")
        
        sample_dir = os.path.join(OUT_ROOT, idx)
        os.makedirs(sample_dir, exist_ok=True)
        
        if os.path.exists(os.path.join(sample_dir, TRACE_FILENAME)):
            print(f"[GPU {rank}] Skip {idx} (Done)")
            continue
            
        if not os.path.exists(img_path): continue

        try:
            working_instr = instr

            session = AgentSession(idx, instr, working_instr, img_path, sample_dir,
                                   routing_class=routing_class)
            final_img = session.run_loop()
            
            print(f"[GPU {rank}] Finished {idx} (class={routing_class})")
            
        except Exception as e:
            print(f"[GPU {rank}] Error {idx}: {e}")
            traceback.print_exc()

def main():
    if not os.path.exists(META_JSON): 
        print(f"[Error] File not found: {META_JSON}")
        return
        
    with open(META_JSON, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
        
    print(f"[*] Found {len(data)} total cases to run.")

    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No GPU found. Exiting.")
        return

    MAX_RESTARTS = 5
    iteration = 1
    
    while True:
        restart_count = iteration - 1
        
        print(f"\n{'='*50}")
        if iteration == 1:
            print("[*] Starting Initial Run (Iteration 1)")
        else:
            print(f"[*] Starting Restart {restart_count} / {MAX_RESTARTS}")
        print(f"{'='*50}")

        clean_count = 0
        pending_ids = []

        for item in data:
            idx = str(item["index"])
            sample_dir = os.path.join(OUT_ROOT, idx)
            trace_path = os.path.join(sample_dir, TRACE_FILENAME)
            
            if os.path.exists(trace_path):
                try:
                    with open(trace_path, 'r', encoding='utf-8') as tf:
                        trace_data = json.load(tf)
                    
                    if trace_data.get("status") not in ("completed", "completed_forced"):
                        shutil.rmtree(sample_dir)
                        clean_count += 1
                        pending_ids.append(idx)
                except Exception:
                    shutil.rmtree(sample_dir)
                    clean_count += 1
                    pending_ids.append(idx)
            else:
                pending_ids.append(idx)
                
        pending_count = len(pending_ids)
        
        print(f"[*] Cleaned {clean_count} corrupted or incomplete traces.")
        print(f"[*] Pending/Unqualified tasks: {pending_count} / {len(data)}")

        if pending_count <= 1:
            print(f"[*] Unqualified count ({pending_count}) is <= 1. Finished successfully!")
            if pending_count == 1:
                print(f"[*] Last incomplete task ID: {pending_ids[0]}")
            break

        if restart_count >= MAX_RESTARTS:
            print(f"[*] Reached maximum restart attempts ({MAX_RESTARTS}). {pending_count} tasks remain incomplete.")
            print(f"[*] Incomplete task IDs: {pending_ids}")
            break

        print(f"[*] Spawning {world_size} GPU workers...")
        mp.spawn(gpu_worker, args=(world_size, data), nprocs=world_size, join=True)
        
        iteration += 1

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError: pass
    main()
