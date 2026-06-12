import os
from typing import List, Dict, Any, Tuple, Union
from PIL import Image, ImageDraw

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def rel1000_to_pixel_bbox(
    bbox_rel1000: Union[List[float], Tuple[float, float, float, float]],
    W: int,
    H: int,
) -> List[int]:
    """
    Convert normalized bbox [0-1000] to pixel coordinates [x0, y0, x1, y1]
    """
    x0, y0, x1, y1 = bbox_rel1000

    x0 = int(round(float(x0) / 1000.0 * W))
    x1 = int(round(float(x1) / 1000.0 * W))
    y0 = int(round(float(y0) / 1000.0 * H))
    y1 = int(round(float(y1) / 1000.0 * H))

    # normalize
    if x1 < x0: x0, x1 = x1, x0
    if y1 < y0: y0, y1 = y1, y0

    # clamp
    x0 = _clamp(x0, 0, W - 1)
    y0 = _clamp(y0, 0, H - 1)
    x1 = _clamp(x1, 1, W)
    y1 = _clamp(y1, 1, H)

    # avoid degenerate (width/height at least 1)
    if x1 <= x0: x1 = _clamp(x0 + 1, 1, W)
    if y1 <= y0: y1 = _clamp(y0 + 1, 1, H)

    return [x0, y0, x1, y1]

def target_tool(
    image_path: str,
    image_id: Union[int, str],
    bbox_rel1000: List[float],
    out_dir: str,
    color: str = "red",
    width: int = 5,
    out_suffix: str = "_target",
    out_ext: str = ".png",
) -> Dict[str, Any]:
    """
    Draw bounding box on image and save.
    
    Args:
        image_path: Input image path
        image_id: Image ID (for output filename)
        bbox_rel1000: [x0, y0, x1, y1] (range 0-1000)
        out_dir: Output directory
        color: Box color (default red)
        width: Box line width (default 5 pixels)
    
    Returns:
        dict: Dict with output_path and metadata
    """
    os.makedirs(out_dir, exist_ok=True)

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"target_tool: Cannot open image {image_path}. Error: {e}")
        
    W, H = img.size

    # Convert bbox from normalized to pixel coordinates
    bbox_px = rel1000_to_pixel_bbox(bbox_rel1000, W, H)
    x0, y0, x1, y1 = bbox_px

    # Draw bounding box
    draw = ImageDraw.Draw(img)
    draw.rectangle([x0, y0, x1, y1], outline=color, width=width)

    # Save result
    out_name = f"{image_id}{out_suffix}{out_ext}"
    out_path = os.path.join(out_dir, out_name)
    img.save(out_path)

    return {
        "image_id": str(image_id),
        "input_path": image_path,
        "output_path": out_path,
        "bbox_rel1000": bbox_rel1000,
        "bbox_px": bbox_px,
        "orig_size": [W, H],
        "draw_params": {"color": color, "width": width}
    }


if __name__ == "__main__":
    pass