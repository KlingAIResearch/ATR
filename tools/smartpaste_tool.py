import os
import numpy as np
from PIL import Image

def _get_bbox_from_alpha(image):
    """
    Helper: Get tight bounding box from alpha channel.
    Remove transparent whitespace around object from SAM3 cutout.
    """
    # Convert to RGBA if needed
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
        
    alpha = np.array(image.split()[-1])
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
    
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    # Returns (left, top, right, bottom)
    return (xmin, ymin, xmax + 1, ymax + 1)

def smartpaste_tool(base_path: str, crop_path: str, target_bbox_rel1000: list, out_dir: str, image_id: str, step_index: int) -> dict:
    """
    Smart paste of transparent cutout to target position.
    
    Logic:
        1. Read cutout from SAM3 (crop_path).
        2. Auto-crop transparent pixels to get actual object size.
        3. Calculate target region from target_bbox_rel1000 (Rel1000 coords).
        4. Preserve aspect ratio and scale to fit target region (Fit mode).
        5. Center paste.
    
    Args:
        base_path (str): Background image path.
        crop_path (str): Transparent cutout path (from sam3_tool).
        target_bbox_rel1000 (list): Target position [x0, y0, x1, y1] (0-1000 range).
        out_dir (str): Output directory.
        image_id (str): Image ID (for naming).
        step_index (int): Step index (for naming).

    Returns:
        dict: {"status": "success"/"error", "output_path": str, "message": str}
    """
    # Basic validation
    if not os.path.exists(base_path):
        return {"status": "error", "message": f"Base image not found: {base_path}"}
    if not os.path.exists(crop_path):
        return {"status": "error", "message": f"Crop image not found: {crop_path}"}
    if not (isinstance(target_bbox_rel1000, list) and len(target_bbox_rel1000) == 4):
        return {"status": "error", "message": f"Invalid target bbox: {target_bbox_rel1000}"}

    try:
        base_img = Image.open(base_path).convert("RGBA")
        crop_img = Image.open(crop_path).convert("RGBA")
        W, H = base_img.size

        # Extract tight content bbox
        content_bbox = _get_bbox_from_alpha(crop_img)
        if content_bbox is None:
            return {"status": "error", "message": "Crop image is completely transparent."}
        
        object_img = crop_img.crop(content_bbox)
        
        # Calculate target region in absolute coordinates
        rel_x0, rel_y0, rel_x1, rel_y1 = target_bbox_rel1000
        
        # Clamp to bounds
        rel_x0, rel_y0 = max(0, rel_x0), max(0, rel_y0)
        rel_x1, rel_y1 = min(1000, rel_x1), min(1000, rel_y1)

        target_x0 = int(rel_x0 / 1000.0 * W)
        target_y0 = int(rel_y0 / 1000.0 * H)
        target_w = int((rel_x1 - rel_x0) / 1000.0 * W)
        target_h = int((rel_y1 - rel_y0) / 1000.0 * H)

        if target_w < 1 or target_h < 1:
            return {"status": "error", "message": "Target bbox resolves to zero size."}

        # Scale with aspect ratio preservation (Fit mode)
        obj_w, obj_h = object_img.size
        ratio = min(target_w / obj_w, target_h / obj_h)
        new_size = (max(1, int(obj_w * ratio)), max(1, int(obj_h * ratio)))
        
        object_resized = object_img.resize(new_size, Image.Resampling.LANCZOS)

        # Center paste to target region
        offset_x = (target_w - new_size[0]) // 2
        offset_y = (target_h - new_size[1]) // 2
        
        paste_x = target_x0 + offset_x
        paste_y = target_y0 + offset_y
        
        # Composite with transparency
        layer = Image.new("RGBA", base_img.size, (0,0,0,0))
        layer.paste(object_resized, (paste_x, paste_y))
        
        out_img_rgba = Image.alpha_composite(base_img, layer)
        out_img = out_img_rgba.convert("RGB")

        # Save result
        os.makedirs(out_dir, exist_ok=True)
        filename = f"{image_id}_step{step_index}_smartpaste.png"
        out_path = os.path.join(out_dir, filename)
        out_img.save(out_path)

        return {
            "status": "success",
            "output_path": out_path,
            "message": "Paste successful"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    pass