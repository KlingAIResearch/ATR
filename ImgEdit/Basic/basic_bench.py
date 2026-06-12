import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor, as_completed

# Google GenAI SDK
try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("Please install the SDK: pip install google-genai")

# Gemini API Configuration
KEY_PATH = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_PROJECT_ID")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("GOOGLE_LOCATION")
GEMINI_MODEL_NAME = "gemini-2.5-pro"

def load_prompts(prompts_json_path):
    with open(prompts_json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_image(image_path):
    """Load image using PIL"""
    try:
        return Image.open(image_path)
    except FileNotFoundError:
        print(f"File {image_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

@retry(wait=wait_exponential(multiplier=1, min=2, max=5), stop=stop_after_attempt(100))
def call_gemini(original_image_path, result_image_path, edit_prompt, edit_type, prompts, client):
    """Call Gemini API for image evaluation"""
    try:
        # Load images
        original_image = load_image(original_image_path)
        result_image = load_image(result_image_path)

        if original_image is None or result_image is None:
            return {"error": "Image loading failed"}

        # Get prompt for the specific edit type
        prompt_template = prompts[edit_type]
        full_prompt = prompt_template.replace('<edit_prompt>', edit_prompt)

        # Create content with text and images
        contents = [
            full_prompt,
            original_image,
            result_image
        ]

        # Call Gemini API
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=contents
        )

        return response.text
    except Exception as e:
        print(f"Error in calling Gemini API: {e}")
        raise

def process_single_item(key, item, result_img_folder, origin_img_root, prompts, client):
    result_img_name = f"{key}.png"
    result_img_path = os.path.join(result_img_folder, result_img_name)
    origin_img_path = os.path.join(origin_img_root, item['id'])
    edit_prompt = item['prompt']
    edit_type = item['edit_type']

    response = call_gemini(origin_img_path, result_img_path, edit_prompt, edit_type, prompts, client)
    return key, response

def process_json(edit_json, result_img_folder, origin_img_root, num_threads, prompts):
    # Set credentials environment variable
    if KEY_PATH:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = KEY_PATH
    
    # Initialize Gemini client
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION
    )
    
    with open(edit_json, 'r', encoding='utf-8') as f:
        edit_infos = json.load(f)

    results = {}
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_key = {
            executor.submit(process_single_item, key, item, result_img_folder, origin_img_root, prompts, client): key
            for key, item in edit_infos.items()
        }

        for future in tqdm(as_completed(future_to_key), total=len(future_to_key), desc="Processing edits"):
            key = future_to_key[future]
            try:
                k, result = future.result()
                results[k] = result
            except Exception as e:
                print(f"Error processing key {key}: {e}")
                results[key] = {"error": str(e)}

    results_path = os.path.join(result_img_folder, 'result.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Evaluate image edits using Gemini")
    parser.add_argument('--result_img_folder', type=str, required=True, help="Folder with subfolders of edited images")
    parser.add_argument('--edit_json', type=str, required=True, help="Path to JSON file mapping keys to metadata")
    parser.add_argument('--origin_img_root', type=str, required=True, help="Root path where original images are stored")
    parser.add_argument('--num_processes', type=int, default=32, help="Number of parallel threads")
    parser.add_argument('--prompts_json', type=str, required=True, help="JSON file containing prompts") 
    args = parser.parse_args()

    print(f"Using Gemini model: {GEMINI_MODEL_NAME}")
    print(f"Project: {PROJECT_ID}, Location: {LOCATION}")
    
    prompts = load_prompts(args.prompts_json)  

    process_json(args.edit_json, args.result_img_folder, args.origin_img_root, args.num_processes, prompts)

if __name__ == "__main__":
    main()
