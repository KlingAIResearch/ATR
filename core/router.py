# ==========================================
# Tool Decision Analyzer - Two-Stage Classification
# ==========================================
# Stage 1: Classify cases into A, B, C categories
# Stage 2: Further classify A category cases into A1, A2
# ==========================================
import os
import json
import time
import threading
import concurrent.futures
from typing import Dict, List

from google import genai
from google.genai import types
from core.runtime_config import create_genai_client, get_gemini_model

# ==========================================
# Configuration
# ==========================================
# Credentials are read from the GOOGLE_APPLICATION_CREDENTIALS environment variable.
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_PROJECT_ID")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("GOOGLE_LOCATION")
GEMINI_MODEL_NAME = get_gemini_model("gemini-3-flash-preview")
MAX_WORKERS = 20


INPUT_JSONL = "./examples/output/data_captions.jsonl"
OUTPUT_STAGE1_JSONL = "./examples/output/stage1_abc_classification.jsonl"
OUTPUT_FINAL_JSONL = "./examples/output/data_decisions_two_stage.jsonl"
OUTPUT_FILTERED_JSONL = "./examples/output/data_decisions_filtered.jsonl"

# ==========================================
# Stage 1: ABC Three-Category Classification Criteria
# ==========================================
STAGE1_PROMPT_ABC = """
You are a multi-modal task routing agent. Based on the provided image caption, edit region analysis, and user instruction, classify the image editing task into exactly ONE of three operational pipelines: A, B, or C.

**Image Caption** (scene description for context):
{caption}

**Edit Region Analysis**:
{edit_region_info}

**User Instruction**:
{instruction}

---

## Classification Rules (A vs B vs C)

Determine which of the three main architectural approaches is required:

**A (Direct Edit)**: 
Use when the task can be handled by a single direct edit call:
- Global scene changes (weather, style, lighting, background swap) and Simple color changes
- Simple Object Removal/Addition: The removal or addition of objects where the background is easy and not complicated
- Texture or appearance style changes applied globally or to clearly identifiable objects
- Pose and Anatomical Changes: Altering the posture, gesture, or structural geometry of humans or animals. 
- Complete Environmental Decoupling: Instructions that require the exclusive preservation of a target entity while completely neutralizing or discarding its surrounding visual context. 

**B (SAM chain: sam3 → qwen_edit → target → smartpaste → refine)**:
Use ONLY for:
- Explicit Spatial Relocation: Instructions that directly command moving or repositioning a target object to a new location within the image.
- Physics-breaking removal: Removing an object that supports or contains another (e.g., removing a chair while someone is sitting on it; removing a shelf while items rest on it)
  **-> EXCEPTION: Do NOT assign to Category B for Physics-breaking removals IF:**
    1. the target object's structural boundaries are visually warped, fragmented, or rendered discontinuous by an intervening transparent/refractive medium or dynamic environmental interference. The SAM-based extraction strictly requires unambiguous, continuous physical boundaries in the 2D image plane to function correctly.
    2. the target object is densely embedded within a complex lattice, mesh, or an assembly of multiple discrete, intersecting components. The SAM-based pipeline struggles to cleanly isolate a single target when it shares extensive, intricate boundaries with numerous other abutting, structurally similar pieces.

**C (Crop chain: crop_tool → qwen_edit → croppaste_tool)**:
Use when precision isolation of a local region is needed:
- Editing a specific named object among multiple similar ones
- Editing multiple distinct targets that are separated
- Changing texture, material, or geometry of a specific part
- Changing outfit, pose, or action of a specific person
- The target is local, clearly bounded, and NOT a whole-scene change
- Avoid using this if the target is a single, highly prominent object 
Do NOT use C for Pose/Action modifications on humans or animals. Cropping through a continuous semantic body part (like a torso) destroys anatomical coherence and results in severe bisection artifacts when pasted back.

### Spatial Threshold Heuristic (A vs C):
When deciding between **A (Direct Diffusion Editing)** and **C (Precision Crop & Paste Chain)**, incorporate the following logic:

**1. Semantic Prominence Overrides Threshold**:
Regardless of the `edit_region_percentage`, if the target object is the single, most prominent, and clearly bounded subject of the image (e.g., "the dog", "the cup"), and the surrounding background is not complex or narrow, prefer Pipeline A.

**2. The < 10% Spatial Threshold Heuristic **:
When choosing between A and C for a small edit (`edit_region_percentage < 10%`):
- **If Background is Complex, Ambiguous, or Visually Critical**: Strongly prefer Pipeline C for isolation and resolution enhancement. This includes cases where the target is surrounded by visually similar or easily confused objects, located in a cluttered region, partially occluded, very small, far away from the camera, or not visually clear enough for reliable direct editing.
- **If Background is Simple**: Prefer Pipeline A. Direct diffusion editing is usually safe when the target is visually clear, isolated, and surrounded by simple or easily reconstructable background.
---

**Output Format (JSON only)**:
{{
  "class": "A" | "B" | "C",
  "reasoning": "One sentence explaining why this pipeline was chosen based on the task's structural and semantic requirements. If edit_region_percentage < 10% and evaluating between A and C, explicitly state how the spatial threshold heuristic influenced the final routing."
}}

Output ONLY the JSON.
"""

# ==========================================
# Stage 2: A-Class Subdivision Criteria
# ==========================================
STAGE2_PROMPT_A1A2 = """
You are an instruction rewriting analyzer. The task has already been classified as **A (Direct Edit)**. Now determine if the instruction needs rewriting.

**Image Caption** (scene description for context):
{caption}

**Edit Region Analysis**:
{edit_region_info}

**User Instruction**:
{instruction}

---

## A-Class Subdivision Rules (A1 vs A2)

Apply the following sequential defect-checking rules to determine if instruction rewriting is needed. 

**Rule 1: Target Under-specification Check (→ A2)**
- **Check**: Does the instruction utilize generic or underspecified terminology that fails to capture the visual complexity or distinguishing features of the target object within the given image context?
- **Logic**: If YES → **A2** (Rewrite needed). The semantic binding between the text and the visual target is insufficient, necessitating augmented visual descriptors to ensure accurate grounding.

**Rule 2: Operational Ambiguity Check (→ A2)**
- **Check**: Does the instruction employ ambiguous editing verbs that lack clear constraints regarding the preservation or modification of the surrounding context and background?
- **Logic**: If YES → **A2** (Rewrite needed). The instruction must be refined to explicitly separate positive constraints (elements to preserve) from negative constraints (elements to modify or remove).

**Rule 3: Inpainting End-State Omission Check (→ A2)**
- **Check**: When the instruction commands the removal of an occluding foreground element or an outer layer, does it fail to specify the semantic content or structural state that should be revealed beneath the removed region?
- **Logic**: If YES → **A2** (Rewrite needed). Executing removal without defining the underlying semantic topology results in uncontrolled over-erasure or structural collapse.

**Rule 4: Abstract Deduction & Indirect Reference Check (→ A2)**
- **Check**: Does the instruction use hypothetical verbs (e.g., "Show", "Imagine", "Make it look like") or require complex visual/logical reasoning to identify the target based on narrative context or indirect relationships (e.g., "the object that stands out", "after a long journey") instead of directly naming the physical object?
- **Logic**: If YES → **A2** (Rewrite needed). The instruction relies on narrative context or cognitive deduction rather than concrete visual grounding, and must be translated into direct, physical editing commands.

**Rule 5: Figure-Ground Separation & Inverse Operation Check (→ A2)**
- **Check**: Does the instruction request abstract operations from a foreground subject without explicitly defining the inverse physical constraints for the background?
- **Logic**: If YES → **A2** (Rewrite needed). Diffusion models do not natively understand digital actions like "extract". These instructions must be explicitly translated into bipartite physical commands: actively altering the background paired with a strict preservation command.

**Default Rule:**
- **Logic**: If NONE of the above rules (1-4) trigger A2 → **A1** (No rewrite needed). The instruction is a direct, unambiguous physical action and is safe for direct execution.

---

**Output Format (JSON only)**:
{{
  "class": "A1" | "A2",
  "reasoning": "One sentence explaining which rule was applied and why A1 or A2 was chosen.",
  "triggered_rule": "Rule 1" | "Rule 2" | "Rule 3" | "Rule 4" | "None (Default to A1)"
}}

Output ONLY the JSON.
"""

# ==========================================
# Helper Functions
# ==========================================
def get_mime_type(image_path: str) -> str:
    """Return MIME type based on file extension"""
    ext = os.path.splitext(image_path)[1].lower()
    return 'image/png' if ext == '.png' else 'image/jpeg'


def extract_caption_info(caption: Dict) -> tuple:
    """Extract caption related information"""
    if isinstance(caption, dict):
        caption_text = caption.get('description', '')
        edit_region_reasoning = caption.get('edit_region_reasoning', 'N/A')
        edit_region_percentage = caption.get('edit_region_percentage', 'N/A')
        edit_region_info = f"Reasoning: {edit_region_reasoning}\nEdit Region Percentage: {edit_region_percentage}%"
    else:
        caption_text = str(caption)
        edit_region_info = "N/A"
    return caption_text, edit_region_info


def call_gemini(prompt: str, image_path: str, client) -> Dict:
    """Call Gemini API"""
    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}
    
    try:
        # Read image
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        
        # Call Gemini
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=[
                prompt,
                types.Part.from_bytes(data=img_bytes, mime_type=get_mime_type(image_path))
            ],
            config=types.GenerateContentConfig(
                max_output_tokens=8192,
                response_mime_type="application/json"
            )
        )
        
        # Parse response
        if response.text is None:
            try:
                decision_text = response.candidates[0].content.parts[0].text.strip()
            except Exception:
                return {"error": "Empty response from Gemini"}
        else:
            decision_text = response.text.strip()

        try:
            result = json.loads(decision_text)
            # Fix: if response is list, take first element
            if isinstance(result, list):
                if len(result) > 0:
                    return result[0]
                else:
                    return {"error": "Empty list response from Gemini"}
            return result
        except json.JSONDecodeError:
            return {"raw_response": decision_text}
            
    except Exception as e:
        return {"error": str(e)}


# ==========================================
# Stage 1: ABC Classification
# ==========================================
def stage1_classify_abc(input_path: str, output_path: str):
    """
    Stage 1: Classify all cases into A, B, C categories
    Supports resume from checkpoint
    """
    print("\n" + "="*60)
    print("Stage 1: ABC Three-Category Classification")
    print("="*60)
    
    # Read input data
    if not os.path.exists(input_path):
        print(f"[Error] File not found: {input_path}")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    print(f"[*] Found {len(data)} data entries")
    
    # Check for saved progress (resume from checkpoint)
    processed_indices = set()
    results = []
    
    if os.path.exists(output_path):
        print(f"[*] Found saved progress file, loading...")
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    results.append(item)
                    processed_indices.add(item['index'])
        print(f"[*] Loaded {len(processed_indices)} processed entries, resuming from checkpoint")
    
    # Statistics
    stats = {"A": 0, "B": 0, "C": 0, "errors": 0}
    
    # Count processed data
    for item in results:
        decision = item.get('stage1_decision', {})
        if 'error' in decision:
            stats["errors"] += 1
        else:
            cls = decision.get('class', '?')
            if cls in stats:
                stats[cls] += 1
    
    # Filter unprocessed data
    unprocessed = [item for item in data if item['index'] not in processed_indices]
    print(f"[*] {len(unprocessed)} entries to process, using {MAX_WORKERS} parallel workers")
    
    lock = threading.Lock()
    completed_count = [0]
    
    def process_stage1_item(item):
        client = create_genai_client()
        image_path = item['input_image']
        instruction = item['instruction']
        caption = item.get('caption', {})
        caption_text, edit_region_info = extract_caption_info(caption)
        prompt = STAGE1_PROMPT_ABC.format(
            caption=caption_text,
            edit_region_info=edit_region_info,
            instruction=instruction
        )
        decision = call_gemini(prompt, image_path, client)
        item['stage1_decision'] = decision
        return item
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(process_stage1_item, item): item for item in unprocessed}
        for future in concurrent.futures.as_completed(future_to_item):
            orig_item = future_to_item[future]
            try:
                result_item = future.result()
                decision = result_item.get('stage1_decision', {})
                with lock:
                    results.append(result_item)
                    completed_count[0] += 1
                    if 'error' in decision:
                        stats["errors"] += 1
                        print(f"  [{completed_count[0]}/{len(unprocessed)}] ❌ {result_item['index']}: {decision['error']}")
                    else:
                        if not isinstance(decision, dict):
                            decision = {"error": f"Invalid response type: {type(decision)}"}
                            result_item['stage1_decision'] = decision
                            stats["errors"] += 1
                        else:
                            cls = decision.get('class', '?')
                            reasoning = decision.get('reasoning', 'N/A')
                            if cls in stats:
                                stats[cls] += 1
                            print(f"  [{completed_count[0]}/{len(unprocessed)}] ✓ {result_item['index']}: [{cls}] {reasoning}")
                    if completed_count[0] % 10 == 0:
                        with open(output_path, 'w', encoding='utf-8') as out_f:
                            for r in results:
                                out_f.write(json.dumps(r, ensure_ascii=False) + '\n')
                        print(f"  💾 Progress saved: {completed_count[0]}/{len(unprocessed)}")
            except Exception as e:
                with lock:
                    orig_item['stage1_decision'] = {"error": str(e)}
                    results.append(orig_item)
                    stats["errors"] += 1
                    completed_count[0] += 1
                    print(f"  [{completed_count[0]}/{len(unprocessed)}] ❌ {orig_item['index']}: {str(e)}")
    
    # Final save
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for result in results:
            out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # Print statistics
    total = len(data)
    print(f"\n" + "="*60)
    print(f"✅ Stage 1 completed! Results saved to: {output_path}")
    
    print(f"\n📊 ABC Classification Statistics:")
    for cls in ["A", "B", "C"]:
        count = stats[cls]
        print(f"  [{cls}]: {count} ({count/total*100:.1f}%)")
    print(f"  [!] Errors: {stats['errors']}")
    
    return results


# ==========================================
# Stage 2: A-Class Subdivision into A1/A2
# ==========================================
def stage2_classify_a1a2(stage1_data: List[Dict]):
    """
    Stage 2: Subdivide A-class cases into A1, A2
    Supports resume from checkpoint
    """
    print("\n" + "="*60)
    print("Stage 2: A-Class Subdivision into A1/A2")
    print("="*60)
    
    # Filter A-class data
    a_class_data = [
        item for item in stage1_data 
        if item.get('stage1_decision', {}).get('class') == 'A'
    ]
    
    print(f"[*] Found {len(a_class_data)} A-class entries to subdivide")
    
    # Check processed data (resume from checkpoint)
    already_processed = [
        item for item in a_class_data
        if 'stage2_decision' in item
    ]
    print(f"[*] {len(already_processed)} already processed, will skip")
    
    # Statistics
    stats = {"A1": 0, "A2": 0, "errors": 0}
    
    # Count processed data
    for item in already_processed:
        decision = item.get('stage2_decision', {})
        if 'error' in decision:
            stats["errors"] += 1
        else:
            cls = decision.get('class', '?')
            if cls in stats:
                stats[cls] += 1
    
    # Filter unprocessed A-class data
    unprocessed_a = [item for item in a_class_data if 'stage2_decision' not in item]
    print(f"[*] {len(unprocessed_a)} A-class entries to process, using {MAX_WORKERS} parallel workers")
    
    lock = threading.Lock()
    completed_count = [0]
    
    def process_stage2_item(item):
        client = create_genai_client()
        image_path = item['input_image']
        instruction = item['instruction']
        caption = item.get('caption', {})
        caption_text, edit_region_info = extract_caption_info(caption)
        prompt = STAGE2_PROMPT_A1A2.format(
            caption=caption_text,
            edit_region_info=edit_region_info,
            instruction=instruction
        )
        decision = call_gemini(prompt, image_path, client)
        item['stage2_decision'] = decision
        return item
    
    # Process each A-class data entry (parallel)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(process_stage2_item, item): item for item in unprocessed_a}
        for future in concurrent.futures.as_completed(future_to_item):
            orig_item = future_to_item[future]
            try:
                result_item = future.result()
                decision = result_item.get('stage2_decision', {})
                with lock:
                    completed_count[0] += 1
                    if 'error' in decision:
                        stats["errors"] += 1
                        print(f"  [{completed_count[0]}/{len(unprocessed_a)}] ❌ {result_item['index']}: {decision['error']}")
                    else:
                        if not isinstance(decision, dict):
                            decision = {"error": f"Invalid response type: {type(decision)}"}
                            result_item['stage2_decision'] = decision
                            stats["errors"] += 1
                        else:
                            cls = decision.get('class', '?')
                            reasoning = decision.get('reasoning', 'N/A')
                            triggered_rule = decision.get('triggered_rule', 'N/A')
                            if cls in stats:
                                stats[cls] += 1
                            print(f"  [{completed_count[0]}/{len(unprocessed_a)}] ✓ {result_item['index']}: [{cls}] {triggered_rule} {reasoning}")
            except Exception as e:
                with lock:
                    orig_item['stage2_decision'] = {"error": str(e)}
                    stats["errors"] += 1
                    completed_count[0] += 1
                    print(f"  [{completed_count[0]}/{len(unprocessed_a)}] ❌ {orig_item['index']}: {str(e)}")
    
    # Print statistics
    total = len(a_class_data)
    print(f"\n" + "="*60)
    print(f"✅ Stage 2 completed!")
    
    print(f"\n📊 A-Class Subdivision Statistics:")
    for cls in ["A1", "A2"]:
        count = stats[cls]
        print(f"  [{cls}]: {count} ({count/total*100:.1f}% of A class)")
    print(f"  [!] Errors: {stats['errors']}")


# ==========================================
# Merge Results
# ==========================================
def merge_results(stage1_data: List[Dict], output_path: str):
    """
    Merge results from both stages to generate final classification
    """
    print("\n" + "="*60)
    print("Merge Results")
    print("="*60)
    
    final_stats = {"A1": 0, "A2": 0, "B": 0, "C": 0, "errors": 0}
    
    for item in stage1_data:
        stage1_cls = item.get('stage1_decision', {}).get('class')
        
        # Build final decision
        final_decision = {}
        
        if stage1_cls == 'A':
            # A-class: use stage 2 results
            stage2_decision = item.get('stage2_decision', {})
            final_cls = stage2_decision.get('class', 'A1')  # Default A1
            
            final_decision = {
                'class': final_cls,
                'stage1_reasoning': item.get('stage1_decision', {}).get('reasoning', 'N/A'),
                'stage2_reasoning': stage2_decision.get('reasoning', 'N/A'),
                'triggered_rule': stage2_decision.get('triggered_rule', 'N/A')
            }
            
            if final_cls in final_stats:
                final_stats[final_cls] += 1
                
        elif stage1_cls in ['B', 'C']:
            # B or C class: use stage 1 results directly
            final_decision = {
                'class': stage1_cls,
                'reasoning': item.get('stage1_decision', {}).get('reasoning', 'N/A')
            }
            
            if stage1_cls in final_stats:
                final_stats[stage1_cls] += 1
        else:
            # Error case
            final_decision = {'error': 'Invalid stage1 classification'}
            final_stats['errors'] += 1
        
        # Add final decision
        item['decision'] = final_decision
        
        # Keep stage decisions for debugging
        # (stage1_decision and stage2_decision already added in previous stages)
    
    # Save final results
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for item in stage1_data:
            out_f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Print statistics
    total = len(stage1_data)
    print(f"\n📊 Final Classification Statistics:")
    for cls in ["A1", "A2", "B", "C"]:
        count = final_stats[cls]
        print(f"  [{cls}]: {count} ({count/total*100:.1f}%)")
    print(f"  [!] Errors: {final_stats['errors']}")
    
    print(f"\n✅ Final results saved to: {output_path}")

    # Additional output: filtered version (keep only index/input_image/instruction/class)
    filtered_path = OUTPUT_FILTERED_JSONL
    with open(filtered_path, 'w', encoding='utf-8') as f:
        for item in stage1_data:
            filtered = {
                "index": item.get("index"),
                "input_image": item.get("input_image"),
                "instruction": item.get("instruction"),
                "class": item.get("decision", {}).get("class"),
            }
            f.write(json.dumps(filtered, ensure_ascii=False) + '\n')
    print(f"✅ Filtered version saved to: {filtered_path}")


# ==========================================
# Main Function
# ==========================================
def main():
    print(f"[*] Input file: {INPUT_JSONL}")
    print(f"[*] Stage 1 output: {OUTPUT_STAGE1_JSONL}")
    print(f"[*] Final output file: {OUTPUT_FINAL_JSONL}")
    print(f"[*] Filtered output file: {OUTPUT_FILTERED_JSONL}")
    
    # Stage 1: ABC classification
    stage1_results = stage1_classify_abc(INPUT_JSONL, OUTPUT_STAGE1_JSONL)
    
    # Stage 2: A-class subdivision
    stage2_classify_a1a2(stage1_results)
    
    # Merge results
    merge_results(stage1_results, OUTPUT_FINAL_JSONL)
    
    print("\n" + "="*60)
    print("🎉 Two-stage classification completed!")
    print("="*60)


if __name__ == "__main__":
    main()
