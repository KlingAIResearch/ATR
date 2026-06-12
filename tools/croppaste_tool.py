import cv2
import numpy as np
import os

def paste_with_seamless_clone(bg_path, fg_path, bbox_rel1000, output_path):
    """
    Paste foreground onto background with hybrid mode:
    - Edge regions: hard paste (direct pixel replacement)
    - Interior regions: Poisson blending for seamless fusion
    """
    
    # 1. Read images
    bg_img = cv2.imread(bg_path, cv2.IMREAD_COLOR)
    fg_img = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)

    if bg_img is None or fg_img is None:
        print(f"[Error] Cannot read images: BG: {bg_path}, FG: {fg_path}")
        return

    bg_h, bg_w = bg_img.shape[:2]

    # 2. Parse coordinates
    x1_rel, y1_rel, x2_rel, y2_rel = bbox_rel1000
    
    # Calculate absolute coordinates
    left = max(0, int((x1_rel / 1000.0) * bg_w))
    top  = max(0, int((y1_rel / 1000.0) * bg_h))
    right = min(bg_w, int((x2_rel / 1000.0) * bg_w))
    bottom = min(bg_h, int((y2_rel / 1000.0) * bg_h))

    target_w = right - left
    target_h = bottom - top

    if target_w <= 0 or target_h <= 0:
        print(f"[Error] Invalid target region: {target_w}x{target_h}")
        return

    # 3. Resize foreground
    fg_resized = cv2.resize(fg_img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    # 4. Decision: check if touching boundary
    is_touching_boundary = (x1_rel <= 0 or y1_rel <= 0 or x2_rel >= 1000 or y2_rel >= 1000)

    # Define hard paste helper function (avoid code duplication)
    def apply_hard_paste():
        # Force RGB only, direct pixel replacement
        if fg_resized.shape[2] == 4:
            fg_rgb = fg_resized[:, :, :3]
        else:
            fg_rgb = fg_resized
        
        bg_img[top:bottom, left:right] = fg_rgb

    if is_touching_boundary:
        # Mode A: Edge hard paste
        apply_hard_paste()
        
    else:
        # Mode B: Interior Poisson blending
        try:
            # Prepare mask
            if fg_resized.shape[2] == 4:
                _, mask = cv2.threshold(fg_resized[:, :, 3], 10, 255, cv2.THRESH_BINARY)
                fg_rgb = fg_resized[:, :, :3]
            else:
                fg_rgb = fg_resized
                mask = 255 * np.ones(fg_rgb.shape, dtype=np.uint8)

            # Calculate center point
            center = (left + target_w // 2, top + target_h // 2)

            # Poisson blending
            result = cv2.seamlessClone(fg_rgb, bg_img, mask, center, cv2.NORMAL_CLONE)
            bg_img = result
            
        except Exception as e:
            apply_hard_paste()

    # 5. Save result
    cv2.imwrite(output_path, bg_img)
    print(f"Output saved to: {output_path}")



if __name__ == "__main__":
    pass