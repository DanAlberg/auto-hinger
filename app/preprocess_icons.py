import os
import cv2
import numpy as np

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
CLEAN_DIR = os.path.join(ASSETS_DIR, "clean")

def strip_white_background(icon_path: str, output_path: str) -> None:
    """Remove white background and save as transparent PNG."""
    img = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"⚠️ Skipping {icon_path} (not found or unreadable)")
        return

    # Convert to grayscale and threshold near-white pixels
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Create alpha channel from mask
    b, g, r = cv2.split(img)
    rgba = cv2.merge([b, g, r, mask])

    # Save cleaned icon
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, rgba)
    print(f"✅ Cleaned: {os.path.basename(icon_path)} → {output_path}")

def preprocess_all_icons():
    """Process all icons in assets/ and save cleaned versions to assets/clean/."""
    os.makedirs(CLEAN_DIR, exist_ok=True)
    for fname in os.listdir(ASSETS_DIR):
        if not fname.lower().endswith(".png"):
            continue
        if fname.startswith("icon_"):
            src = os.path.join(ASSETS_DIR, fname)
            dst = os.path.join(CLEAN_DIR, fname)
            if os.path.exists(dst):
                print(f"⏩ Skipping {fname} (already cleaned)")
                continue
            strip_white_background(src, dst)

if __name__ == "__main__":
    preprocess_all_icons()
