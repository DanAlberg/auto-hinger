from helper_functions import capture_screenshot, swipe, get_screen_resolution
from cv_biometrics import find_biometrics_band_y, extract_biometrics_from_carousel
import time


def main():
    from helper_functions import connect_device
    device = connect_device()
    W, H = get_screen_resolution(device)
    print(f"Screen resolution: {W}x{H}")

    # Start from top of profile
    print("Scrolling from top until Y-band detected...")
    y_found = None
    for i in range(8):
        shot = capture_screenshot(device, f"scroll_{i}")
        y_info = find_biometrics_band_y(shot)
        if y_info.get("found"):
            y_found = y_info["y"]
            print(f"✅ Found Y-band at y={y_found} (method={y_info['method']})")
            break
        swipe(device, int(W * 0.5), int(H * 0.8), int(W * 0.5), int(H * 0.2), duration=600)
        time.sleep(1.5)

    if not y_found:
        print("❌ Failed to detect Y-band after scrolling.")
        return

    print("Running horizontal biometrics extraction...")
    result = extract_biometrics_from_carousel(device, y_override=y_found)
    print("✅ Extraction complete.")
    print(result["biometrics"])


import shutil
import os

# Clear images and debug folders before each run
def clear_output_dirs():
    for folder in ["images", os.path.join("images", "debug")]:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"🧹 Cleared folder: {folder}")
            except Exception as e:
                print(f"⚠️ Could not clear {folder}: {e}")
        os.makedirs(folder, exist_ok=True)

clear_output_dirs()

if __name__ == "__main__":
    main()
