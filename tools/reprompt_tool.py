import os
import sys
import json
import traceback
from PIL import Image

# Google GenAI SDK
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("[RefinementTool] Warning: google-genai SDK not installed.")

class RefinementGenerator:
    def __init__(self, 
                 project_id=None, 
                 location=None, 
                 model_name="gemini-3-flash-preview",
                 key_path=None):
        """
        RefinementGenerator: Read original image, current partial result, and original instruction,
        then generate refinement instruction.
        """
        self.model_name = model_name
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_PROJECT_ID")
        self.location = location or os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("GOOGLE_LOCATION")
        
        key_path = key_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if key_path and os.path.exists(key_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_path

        try:
            self.client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location
            )
        except Exception as e:
            print(f"[Refinement] Init failed: {e}")
            self.client = None

    def _clean_json_text(self, text):
        text = text.strip()
        if text.startswith("```json"): text = text[7:]
        if text.startswith("```"): text = text[3:]
        if text.endswith("```"): text = text[:-3]
        return text.strip()

    def generate_instruction(self, original_path: str, current_path: str, original_instruction: str) -> dict:
        """
        Args:
            original_path: Original background image
            current_path: Current processed image (85% done)
            original_instruction: User's original instruction
        """
        if not self.client:
            return {"status": "error", "error": "Client not initialized"}
        if not os.path.exists(original_path) or not os.path.exists(current_path):
            return {"status": "error", "error": "Images not found"}

        try:
            # 1. Load images
            img_original = Image.open(original_path).convert("RGB")
            img_current = Image.open(current_path).convert("RGB")

            # 2. Build system prompt
            sys_prompt = """
            You are an expert Image Editor Refiner.
            
            **Context:**
            1. **[Original Image]**: The starting point.
            2. **[Original Instruction]**: The user asked to edit the image (e.g., "Move the ladder").
            3. **[Current Edit (85% Done)]**: A mechanical cut-and-paste result of that instruction. The object is in the NEW correct position, but it may looks "fake".

            **Your Task:**
            Generate a **follow-up instruction** for Qwen-Edit to finalize this image.
            
            **Critical Constraints:**
            1. **DO NOT MOVE IT AGAIN**: The "Move" instruction has already been executed. Do NOT say "Move the xxx" again.
            2. **FIX THE ARTIFACTS**: Focus purely on making the object look REAL in its *current* new position.
            3. **DETAILS**: Look for truncated parts (feet) and missing shadows.
            4. **LENGTH**: Keep it under 20 words. Concise English.

            **Output Format (JSON):**
            {
                "analysis": "Identify what looks fake (e.g. 'Ladder moved left, but shadow missing').",
                "qwen_instruction": "The short refinement prompt."
            }
            """

            # 3. Build input content
            prompt_content = [
                sys_prompt,
                img_original, "Image 1: Original Input",
                f"User's Original Instruction: \"{original_instruction}\"",
                img_current,  "Image 2: Current Result (Object moved, needs refinement)",
                "Task: Generate the refinement instruction."
            ]

            # 4. Call Gemini
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt_content,
                config=types.GenerateContentConfig(
                    temperature=0.2, 
                    response_mime_type="application/json"
                )
            )

            # 5. Parse result
            clean_text = self._clean_json_text(response.text)
            try:
                result_json = json.loads(clean_text)
            except json.JSONDecodeError:
                 result_json = {"qwen_instruction": original_instruction, "analysis": "Error parsing JSON"}
            
            instruction = result_json.get("qwen_instruction", "")
            analysis = result_json.get("analysis", "")

            # Length limit
            if len(instruction.split()) > 25:
                 instruction = " ".join(instruction.split()[:20])

            return {
                "status": "success",
                "instruction": instruction,
                "analysis": analysis
            }

        except Exception as e:
            print(f"[Refinement] Error: {e}")
            traceback.print_exc()

            fallback = "Fix the edited object, restore missing parts and add a realistic shadow."
            return {"status": "error", "instruction": fallback}

# ==========================================
# Tool Function Wrapper
# ==========================================
def reprompt_tool(original_path: str, current_path: str, original_instruction: str, model_name="gemini-3-flash-preview") -> dict:
    """
    Wrapper function for calling from Planner/Executor.
    """
    generator = RefinementGenerator(model_name=model_name)
    return generator.generate_instruction(original_path, current_path, original_instruction)


if __name__ == "__main__":
    pass
