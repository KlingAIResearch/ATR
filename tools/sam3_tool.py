import sys
import os
import torch
import numpy as np
from PIL import Image

# ================= Configuration =================

# Root of the GitHub SAM3 code repo. This directory must contain the Python
# package directory named "sam3" so imports like "from sam3.model_builder" work.
CODE_ROOT = os.environ.get("ATR_SAM3_DIR", "./examples/sam3_repo")

# SAM3 checkpoint file downloaded from ModelScope.
CHECKPOINT_PATH = os.environ.get("ATR_SAM3_CHECKPOINT", "./examples/sam3_model/sam3.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =================================================

# Inject path for SAM3 module import
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

# Global cache for model to avoid reloading
_SAM3_MODEL = None
_SAM3_PROCESSOR = None

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    print(f"[SAM3 Tool] Import failed: {e}")

# ================= Helper Functions =================

def _load_sam3_model():
    """Lazy load model on first call only"""
    global _SAM3_MODEL, _SAM3_PROCESSOR
    
    if _SAM3_MODEL is not None and _SAM3_PROCESSOR is not None:
        return _SAM3_MODEL, _SAM3_PROCESSOR

    print("[SAM3 Tool] Initializing SAM 3 model...")
    
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"[SAM3 Tool] Checkpoint not found: {CHECKPOINT_PATH}")

    try:
        # Try to build model
        try:
            model = build_sam3_image_model(model_type="sam3_hiera_large")
        except TypeError:
            print("[SAM3 Tool] model_type parameter not supported, building without parameter...")
            model = build_sam3_image_model()
            
        print(f"[SAM3 Tool] Loading checkpoint: {CHECKPOINT_PATH}")
        state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
        
        # Handle nested state dict
        if "model" in state_dict: state_dict = state_dict["model"]
        elif "state_dict" in state_dict: state_dict = state_dict["state_dict"]
        
        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE).eval()
        
        processor = Sam3Processor(model)
        
        _SAM3_MODEL = model
        _SAM3_PROCESSOR = processor
        print("[SAM3 Tool] Model loaded successfully!")
        return model, processor
        
    except Exception as e:
        print(f"[SAM3 Tool] Model loading failed: {e}")
        raise e

def _get_bbox_from_mask(mask):
    """Calculate bounding box (xyxy) from binary mask"""
    if isinstance(mask, torch.Tensor): mask = mask.cpu().numpy()
    if mask.ndim == 3: mask = mask[0]
    
    y_indices, x_indices = np.where(mask > 0)
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None

    x0 = np.min(x_indices)
    x1 = np.max(x_indices)
    y0 = np.min(y_indices)
    y1 = np.max(y_indices)
    
    return [int(x0), int(y0), int(x1), int(y1)]

def _save_cutout(image, mask, save_path):
    """Save cutout with transparent background"""
    if isinstance(mask, torch.Tensor): mask = mask.cpu().numpy()
    if mask.ndim == 3: mask = mask[0]

    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask_uint8, mode="L")
    image_rgba = image.convert("RGBA")
    image_rgba.putalpha(mask_pil)
    image_rgba.save(save_path)

def _apply_mask_overlay(image, mask, color, alpha=0.5):
    """Apply color overlay for visualization"""
    if isinstance(mask, torch.Tensor): mask = mask.cpu().numpy()
    if mask.ndim == 3: mask = mask[0]
    if not (mask > 0).any(): return image
    
    overlay = Image.new("RGBA", image.size, color + (0,))
    mask_img = Image.fromarray(((mask > 0) * 255).astype(np.uint8), mode="L")
    solid = Image.new("RGBA", image.size, color + (int(255 * alpha),))
    image = image.convert("RGBA")
    image.paste(solid, (0, 0), mask_img)
    return image.convert("RGB")

# ================= Core Tool Functions ==================

def sam3_tool(image_path: str, text_prompts, output_dir: str = None) -> dict:
    """
    SAM3 segmentation tool.
    
    Args:
        image_path (str): Input image path.
        text_prompts (str or list):
            - Single target: pass string (e.g., "vase") or single-element list, returns highest confidence instance mask.
            - Multi target: pass multi-element list (e.g., ["vase", "lamp"]), returns logical OR of all valid masks.
        output_dir (str, optional): Output directory for results.
    """
    if not os.path.exists(image_path):
        return {"status": "error", "message": f"Image not found: {image_path}"}

    # Normalize prompt format: ensure it's a list
    if isinstance(text_prompts, str):
        text_prompts = [text_prompts]

    is_single = len(text_prompts) == 1

    if output_dir is None:
        base_dir = os.path.dirname(image_path)
        output_dir = os.path.join(base_dir, "sam3_outputs")
    os.makedirs(output_dir, exist_ok=True)

    try:
        model, processor = _load_sam3_model()
        raw_image = Image.open(image_path).convert("RGB")
        width, height = raw_image.size

        mode_label = "Single target (highest confidence)" if is_single else "Multi target (logical OR)"
        print(f"[SAM3 Tool] Segmenting: {mode_label} | Prompts: {text_prompts}")

        with torch.no_grad():
            inference_state = processor.set_image(raw_image)

            if is_single:
                # ---- Single target: get highest confidence instance ----
                output = processor.set_text_prompt(
                    state=inference_state,
                    prompt=text_prompts[0]
                )
                masks = output["masks"]
                scores = output["scores"]

                if len(masks) == 0:
                    return {"status": "error", "message": f"No masks found for prompt: '{text_prompts[0]}'"}

                best_idx = scores.argmax().item()
                best_mask = (masks[best_idx] > 0).float()
                best_score = scores[best_idx].item()

            else:
                # ---- Multi target: threshold filter and logical OR merge ----
                all_valid_masks = []
                all_valid_scores = []
                score_threshold = 0.25

                for prompt in text_prompts:
                    output = processor.set_text_prompt(
                        state=inference_state,
                        prompt=prompt
                    )
                    masks = output["masks"]
                    scores = output["scores"]

                    if len(masks) == 0:
                        continue

                    valid_indices = torch.where(scores > score_threshold)[0]
                    if len(valid_indices) > 0:
                        all_valid_masks.append(masks[valid_indices])
                        all_valid_scores.append(scores[valid_indices])

                if len(all_valid_masks) == 0:
                    return {"status": "error", "message": "No masks found for any prompts."}

                combined_masks = torch.cat(all_valid_masks, dim=0)
                combined_scores = torch.cat(all_valid_scores, dim=0)
                # Logical OR: keep pixels recognized by any prompt
                best_mask = torch.any(combined_masks > 0, dim=0).float()
                best_score = combined_scores.mean().item()

        # Calculate bounding box
        bbox = _get_bbox_from_mask(best_mask)
        bbox_rel1000 = None
        if bbox:
            x0, y0, x1, y1 = bbox
            bbox_rel1000 = [
                max(0, min(1000, int(round(x0 / width * 1000)))),
                max(0, min(1000, int(round(y0 / height * 1000)))),
                max(0, min(1000, int(round(x1 / width * 1000)))),
                max(0, min(1000, int(round(y1 / height * 1000))))
            ]

        # Save outputs (filename from prompts, truncated length)
        safe_prompt = "_and_".join([p.strip().replace(" ", "_") for p in text_prompts])[:50]

        cutout_path = os.path.join(output_dir, f"cutout_{safe_prompt}.png")
        _save_cutout(raw_image, best_mask, cutout_path)

        vis_path = os.path.join(output_dir, f"vis_{safe_prompt}.png")
        vis_img = _apply_mask_overlay(raw_image.copy(), best_mask, (255, 0, 0))
        vis_img.save(vis_path)

        print(f"[SAM3 Tool] Segmentation complete! BBox Rel1000: {bbox_rel1000}, Score: {best_score:.3f}")

        return {
            "status": "success",
            "best_mask_info": {
                "bbox_raw": bbox,
                "bbox_rel1000": bbox_rel1000,
                "cutout_path": cutout_path,
                "score": best_score
            },
            "visualization_path": vis_path
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    pass
