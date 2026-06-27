# qwenv3/tools_v1/ifinish_tool.py
import os
import json
from PIL import Image
from google import genai
from google.genai import types
from core.runtime_config import create_genai_client

def ifinish_tool(original_image_path: str, current_image_path: str, instruction: str, project_id: str, location: str, model_name: str) -> dict:
    """
    Evaluate if image editing task has been completed globally.
    Returns: {"status": "success"/"error", "is_finished": true/false, "reason": str}
    """
    if not os.path.exists(original_image_path) or not os.path.exists(current_image_path):
        return {"status": "error", "message": "Input image path not found."}

    client = create_genai_client()
    
    img_orig = Image.open(original_image_path).convert("RGB")
    img_curr = Image.open(current_image_path).convert("RGB")

    sys_prompt = f"""
    You are a strict QA inspector evaluating the FINAL result of an automated image editing pipeline.
    **Original User Instruction**: "{instruction}"
    
    Your task is to compare the 'Original Image' with the 'Current Image'.
    Has the MAIN objective of the instruction been successfully achieved in the 'Current Image'? 
    
    - DO NOT be overly pedantic about minor lighting or shadow artifacts.
    - FOCUS ONLY on the primary goal (e.g., Is the object removed? Is the color changed? Is it moved to the correct area?).
    - If the main task is done, return "is_finished": true.
    **SPECIAL FOR "EXTRACT" OR "ISOLATE"**: 
        - If the user instruction asks to "Extract" or "Isolate" an object, the CORRECT and EXPECTED pipeline behavior is to **remove the background (e.g., replace it with a solid white or clean background)** while keeping the target object intact.
        - DO NOT fail the evaluation if the original background is missing.
        - If the target object is successfully isolated on a clean/white background, you MUST return "is_finished": true.
    Output format (JSON Only):
    {{
        "is_finished": true or false,
        "reason": "Brief explanation of what is achieved or what is still missing"
    }}
    """

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[sys_prompt, img_orig, "Input: Original Image", img_curr, "Input: Current Image"],
            config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
        )
        res_data = json.loads(response.text.strip())
        return {
            "status": "success", 
            "is_finished": res_data.get("is_finished", False), 
            "reason": res_data.get("reason", "")
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "is_finished": False, "reason": "API Error"}
