import os
import io
from PIL import Image
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

# ==========================================
# System Prompt with Few-Shot Examples
# ==========================================
SYSTEM_PROMPT = """
You are an expert Prompt Evaluator and Refiner for an AI Image Editing System.
Your task is to first evaluate the user's raw instruction, and then either output it UNCHANGED (if it's already good) or REWRITE it (if it's vague or complex).

### STEP 1: EVALUATION CRITERIA
**Condition A: DO NOT REWRITE (Output the exact original instruction)**
- The instruction is already a direct, clear, and unambiguous physical action.
- It directly names the object without requiring complex reasoning.
- It does NOT contain narrative fluff.

**Condition B: MUST REWRITE (Apply the Rewrite Rules)**
- The instruction uses hypothetical or abstract verbs (e.g., "Show", "Imagine").
- The instruction relies on complex visual reasoning to identify the target (e.g., "the object that stands out", "the one used for sitting").
- The instruction contains narrative context (e.g., "after a long journey", "as if trying to").

### STEP 2: REWRITE RULES (Only if Condition B applies)
1. **De-contextualize**: Remove "why", "because", "after a long journey", "as if trying to...", "to show the effect of...". Focus strictly on the VISUAL RESULT.
2. **Physical Actions**: Use imperative verbs: "Add", "Remove", "Replace", "Make [object] [state]". Avoid "Show what happens if...".
3. **Absolute Localization**: Use explicit spatial terms like "on the far right", "leftmost", "top", "bottom". Avoid relative descriptions like "to the right of the sign" unless necessary.
4. **Simplify Objects**: Don't use complex names like "two-wheeled urban transport vehicle". Use "wheels" or "cart".
5. **Concrete Visuals**: Convert abstract states ("exhausted") into visual traits ("hunched over"). Convert abstract effects ("water effect") into physical elements ("water drops").
6. **Constraints**: If removing an object X that supports object Y, explicitly state "keep object Y".

### FEW-SHOT EXAMPLES :

--- Examples of Condition A (DIRECT PASS-THROUGH)---
Input Instruction: "Fold the top towel into the shape of a small boat"
Refined Prompt: "Fold the top towel into the shape of a small boat"

Input Instruction: "Remove the water from the glass."
Refined Prompt: "Remove the water from the glass."

Input Instruction: "Change the red car on the left to blue."
Refined Prompt: "Change the red car on the left to blue."

--- Examples of Condition B (REWRITING NEEDED) ---
Input Instruction: "Make the bird that is intently looking towards the left side of the cage tap its beak against the wire mesh as if trying to get attention."
Refined Prompt: "Make the bird on the far right tap the wire mesh with its beak."

Input Instruction: "Remove the magazine serving as a base for the banana, making sure the area beneath it is seamlessly filled in."
Refined Prompt: "Remove the magazine below the banana, while keep the banana on the table."

Input Instruction: "Replace the banana that stands out because it retains its color amidst the desaturated fruit with a vibrantly colored dragon fruit."
Refined Prompt: "Replace the single vibrant banana on the far right with a pitaya."

Input Instruction: "Imagine the two-wheeled urban transport vehicle has just received a flat tire and is now disabled."
Refined Prompt: "Make the wheels in the picture damaged."

Input Instruction: "Show what the person who is using a walking aid will look like after a long journey."
Refined Prompt: "Make the man on the most right look exhausted and hunched over."

Input Instruction: "Show what would happen if a dark drink were spilled on the fully extended lounge chair."
Refined Prompt: "Add some dark drink drops to the fully extended lounge chair."

Input Instruction: "Show the boat to the right of the 'NO WAKE' sign creating a ripple as if it's just started moving in the water."
Refined Prompt: "Add a ripple to the most right boat as if it's just started moving in the water."

Input Instruction: "Show the effect of water dripping on the blue bicycle helmet from the tree above."
Refined Prompt: "Add water dripping from the tree above to the left blue bicycle."

### TASK:
Analyze the provided image and the user's raw instruction. Output ONLY the final prompt string (either the original if it meets Condition A, or the rewritten version if it meets Condition B). Do not output any explanations or labels.
"""

def fixprompt_tool(image_path: str, instruction: str) -> str:
    """
    Uses Gemini to refine an image editing prompt based on the visual context.
    
    Args:
        image_path (str): Path to the local image file.
        instruction (str): The original, raw user instruction.
        
    Returns:
        str: The refined, precise prompt.
    """
    try:
        # 1. Prepare Image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        with open(image_path, 'rb') as f:
            img_bytes = f.read()

        # 2. Initialize Client
        client = create_genai_client()

        # 3. Call Gemini
        # We pass the System Prompt + Image + User Instruction
        # [Fix] Using types.Part constructors for compatibility
        prompt_parts = [
            types.Part(text=SYSTEM_PROMPT),
            types.Part(inline_data=types.Blob(
                mime_type='image/jpeg',
                data=img_bytes
            )),
            types.Part(text=f"Raw Instruction: {instruction}\nRefined Prompt:")
        ]

        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=[types.Content(role="user", parts=prompt_parts)],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1024
            )
        )

        # 4. Extract Result
        if not response.text:
            return instruction

        refined_prompt = response.text.strip()
        
        # Clean up if the model outputs quotes or extra spacing
        refined_prompt = refined_prompt.replace('"', '').replace("'", "")
        
        return refined_prompt

    except Exception as e:
        print(f"[FixPrompt Error] Could not refine prompt: {e}")
        # Fallback: If API fails, return the original instruction so the pipeline doesn't break
        return instruction

# ==========================================
# Quick Test (Optional)
# ==========================================
if __name__ == "__main__":
    # You can test it directly if you have an image path
    import sys
    if len(sys.argv) > 2:
        img = sys.argv[1]
        instr = sys.argv[2]
        print(f"Original: {instr}")
        print(f"Refined : {fixprompt_tool(img, instr)}")
