# ==========================================
# Image Caption Generator
# ==========================================
# Generate detailed descriptions for images for subsequent Planner decision-making
import os
import json
import time
import sys
from typing import List, Dict

from google import genai
from google.genai import types

# ==========================================
# Configuration
# ==========================================
# Credentials are read from the GOOGLE_APPLICATION_CREDENTIALS environment variable.
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_PROJECT_ID")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("GOOGLE_LOCATION")
GEMINI_MODEL_NAME = "gemini-3-flash-preview"

INPUT_JSONL = "./examples/data/input.jsonl"
OUTPUT_JSONL = "./examples/output/data_captions.jsonl"

# ==========================================
# Caption System Prompt
# ==========================================
CAPTION_SYSTEM_PROMPT = """
You are an expert image analyst. Describe the image naturally, focusing on what the user wants to edit, and estimate the size of the editing region.

**Output Format (JSON)**:
{
  "description": "Natural paragraph describing the scene, with more detail on the edit target and less detail on background.",
  "edit_region_reasoning": "Brief reasoning about the size of the editing region based on the instruction and image",
  "edit_region_percentage": "Percentage of the total image area that will be edited (e.g., 25.0 for 25%)"
}

**Writing Style**:
- Write naturally, as if explaining to someone what you see
- Put more detail on what needs to be edited, less on distant background

**Edit Region Estimation**:
- Assume the entire image is 1000x1000 in relative coordinates
- First, analyze the instruction and identify what region needs to be edited
- Provide reasoning: describe which parts of the image will be affected and estimate their dimensions
- Then calculate the percentage: (estimated_width × estimated_height) / (1000 × 1000) × 100
- For global changes (e.g., "change weather", "change background"), use close to 100%
- For local object edits, estimate the bounding area of that object
- For multiple similar objects where one is specified (e.g., "the middle cup"), estimate just that object's area

**Example 1** (Local object removal):
Instruction: "Remove the middle bucket"

```json
{
  "description": "A wooden bucket sits at the center of a rough table, slightly tilted. Two identical buckets flank it on either side, which could cause confusion. The setting appears to be a rustic workshop with shelves and tools visible in the background.",
  "edit_region_reasoning": "The middle bucket is a single, medium-sized object positioned in the center. Including some surrounding area for inpainting, it occupies roughly 300x400 pixels in a 1000x1000 reference frame.",
  "edit_region_percentage": 12.0
}
```

**Example 2** (Global scene change):
Instruction: "Change the scene to winter"

```json
{
  "description": "A lush green forest with dense foliage and bright sunlight filtering through the trees. The ground is covered with grass and scattered leaves.",
  "edit_region_reasoning": "This is a global scene transformation affecting the entire image - weather, lighting, and environmental elements need to change throughout.",
  "edit_region_percentage": 100.0
}
```

**Example 3** (Background replacement):
Instruction: "Change the water and greenery background to a snowy forest environment"

```json
{
  "description": "A bear stands in the foreground on a weathered log, surrounded by water and lush greenery. The background occupies most of the visible area.",
  "edit_region_reasoning": "The bear and log in the foreground should remain, but the water and greenery background that surrounds them covers approximately 800x700 pixels of the image.",
  "edit_region_percentage": 56.0
}
```

**Key Points**:
- Natural, flowing language for description
- Detail gradient: more on edit subject, less on background
- 50 words maximum for description
- Provide clear reasoning about the editing region before stating the percentage
- Be realistic in size estimation: consider the actual proportion of the editing target in the image

Output ONLY the JSON.
""".strip()

# ==========================================
# Caption Generation Function
# ==========================================
def generate_caption(image_path: str, instruction: str, client) -> Dict:
    """
    Generate structured description for a single image (focused on editing target)
    """
    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}
    
    try:
        # Read image
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        
        # Build targeted prompt
        targeted_prompt = (
            f"The user wants to: \"{instruction}\"\n\n"
            f"Focus on the TARGET OBJECT mentioned in the instruction and its surrounding area. "
            f"Provide the structured description in JSON format."
        )
        
        # Call Gemini
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=[
                CAPTION_SYSTEM_PROMPT,
                targeted_prompt,
                types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg')
            ],
            config=types.GenerateContentConfig(
                max_output_tokens=8196,
                response_mime_type="application/json"  # Force JSON output
            )
        )
        
        # Parse response
        caption_text = response.text.strip()
        
        # Try to parse JSON
        try:
            caption_data = json.loads(caption_text)
            return caption_data
        except json.JSONDecodeError:
            # If not standard JSON, return raw text
            return {"raw_caption": caption_text}
            
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# Batch Processing
# ==========================================
def process_dataset(input_path: str, output_path: str):
    """
    Process the entire dataset
    """
    # Initialize Gemini client
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    
    # Read input data
    if not os.path.exists(input_path):
        print(f"[Error] File not found: {input_path}")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    print(f"[*] Found {len(data)} data entries")
    
    # Process each data entry
    results = []
    for idx, item in enumerate(data):
        print(f"[{idx+1}/{len(data)}] Processing: {item['index']}")
        
        image_path = item['input_image']
        instruction = item['instruction']
        
        # Generate caption (focused on target object)
        caption = generate_caption(image_path, instruction, client)
        
        # Merge into original data
        item['caption'] = caption
        results.append(item)
        
        # Print result preview
        if 'error' in caption:
            print(f"  ❌ Error: {caption['error']}")
        else:
            desc = caption.get('description', 'N/A')
            reasoning = caption.get('edit_region_reasoning', 'N/A')
            region_pct = caption.get('edit_region_percentage', 'N/A')
            # Print description and region info
            print(f"  ✓ Description: {desc[:60]}...")
            print(f"  ✓ Region reasoning: {reasoning[:70]}...")
            print(f"  ✓ Edit region percentage: {region_pct}%")
        
        # Save every 10 entries (prevent loss on interruption)
        if (idx + 1) % 10 == 0:
            with open(output_path, 'w', encoding='utf-8') as out_f:
                for result in results:
                    out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"  💾 Progress saved: {idx+1}/{len(data)}")
        
        # Avoid API rate limiting
        time.sleep(0.5)
    
    # Final save
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for result in results:
            out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Completed! Results saved to: {output_path}")

# ==========================================
# Main Function
# ==========================================
def main():
    print(f"[*] Input file: {INPUT_JSONL}")
    print(f"[*] Output file: {OUTPUT_JSONL}")
    
    process_dataset(INPUT_JSONL, OUTPUT_JSONL)

if __name__ == "__main__":
    main()
