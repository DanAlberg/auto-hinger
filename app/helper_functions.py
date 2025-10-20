from ppadb.client import Client as AdbClient
import time
import cv2
import numpy as np
import os
import glob
import json
from typing import Sequence, Union, List, Tuple, Dict, Any


def _should_emit_cv_debug() -> bool:
    """Gate additional per-template CV overlays by env var set from config (scrape-only/verbose)."""
    try:
        return os.environ.get("HINGE_CV_DEBUG_MODE", "0") == "1"
    except Exception:
        return False



def clear_screenshots_directory():
    """
    Clear all old screenshots from the images directory to prevent confusion
    """
    try:
        if os.path.exists("images"):
            # Remove all PNG files in the images directory
            old_screenshots = glob.glob("images/*.png")
            count = len(old_screenshots)
            
            if count > 0:
                print(f"🗑️  Clearing {count} old screenshots from images directory...")
                for screenshot in old_screenshots:
                    os.remove(screenshot)
                print("✅ Screenshots directory cleared")
            else:
                print("📁 Images directory already clean")
        else:
            print("📁 Images directory doesn't exist - will be created when needed")
            
    except Exception as e:
        print(f"⚠️  Warning: Could not clear screenshots directory: {e}")


# Use to connect directly
def connect_device(user_ip_address="127.0.0.1"):
    adb = AdbClient(host=user_ip_address, port=5037)
    devices = adb.devices()

    print("Devices connected: ", devices)

    if len(devices) == 0:
        print("No devices connected")
        return None
    device = devices[0]
    print(f"Connected to {device.serial}")
    return device


def capture_screenshot(device, filename):
    """
    Capture screenshot with timestamp to prevent confusion between screenshots
    """
    timestamp = int(time.time() * 1000)  # millisecond timestamp
    
    result = device.screencap()
    # Ensure images directory exists
    os.makedirs("images", exist_ok=True)
    
    # Add timestamp to filename for uniqueness
    timestamped_filename = f"{timestamp}_{filename}.png"
    filepath = f"images/{timestamped_filename}"
    
    with open(filepath, "wb") as fp:
        fp.write(result)
    
    print(f"📸 Screenshot saved: {filepath}")
    return filepath


def tap(device, x, y):
    """Basic tap function"""
    device.shell(f"input tap {x} {y}")


def tap_with_confidence(device, x, y, confidence=1.0, tap_area_size="medium"):
    """
    Enhanced tap function with accuracy adjustments based on confidence and area size
    """
    # Adjust tap position based on confidence and area size
    if confidence < 0.7:
        # If low confidence, tap slightly offset to increase hit chance
        offset = 20 if tap_area_size == "small" else 10
        device.shell(f"input tap {x - offset} {y}")
        time.sleep(0.2)
        device.shell(f"input tap {x + offset} {y}")
    elif tap_area_size == "large":
        # For large areas, tap the center
        device.shell(f"input tap {x} {y}")
    else:
        # Standard tap
        device.shell(f"input tap {x} {y}")
    
    print(f"Tapped at ({x}, {y}) with confidence {confidence:.2f}")


def dismiss_keyboard(device, width=None, height=None):
    """
    Try multiple methods to dismiss/hide the on-screen keyboard
    
    Returns:
        bool: True if likely successful, False otherwise
    """
    methods_tried = []
    
    try:
        # Method 1: Press Enter (might send message in some apps)
        print("  � Trying ENTER key to close keyboard...")
        device.shell("input keyevent KEYCODE_ENTER")
        methods_tried.append("ENTER")
        time.sleep(1)
        
    except Exception as e:
        print(f"  ⚠️  ENTER key failed: {e}")
    
    try:
        # Method 2: Back key to hide keyboard
        print("  ⬅️  Trying BACK key to hide keyboard...")
        device.shell("input keyevent KEYCODE_BACK")
        methods_tried.append("BACK")
        time.sleep(1)
        
    except Exception as e:
        print(f"  ⚠️  BACK key failed: {e}")
    
    try:
        # Method 3: Hide keyboard ADB command
        print("  📱 Trying hide keyboard command...")
        device.shell("ime disable com.android.inputmethod.latin/.LatinIME")
        time.sleep(0.5)
        device.shell("ime enable com.android.inputmethod.latin/.LatinIME")
        methods_tried.append("IME_TOGGLE")
        time.sleep(1)
        
    except Exception as e:
        print(f"  ⚠️  IME toggle failed: {e}")
    
    try:
        # Method 4: Tap outside keyboard area
        if width and height:
            print("  👆 Trying tap outside keyboard area...")
            # Tap in upper third of screen where keyboard shouldn't be
            tap(device, int(width * 0.5), int(height * 0.25))
            methods_tried.append("TAP_OUTSIDE")
            time.sleep(1)
            
    except Exception as e:
        print(f"  ⚠️  Tap outside failed: {e}")
    
    print(f"  📝 Keyboard dismissal methods tried: {', '.join(methods_tried)}")
    return len(methods_tried) > 0


def input_text(device, text):
    # Escape spaces in the text
    text = text.replace(" ", "%s")
    print("text to be written: ", text)
    device.shell(f'input text "{text}"')


def input_text_robust(device, text, max_attempts=3):
    """
    Robust text input with multiple methods and verification
    
    Args:
        device: ADB device object
        text: Text to input
        max_attempts: Maximum retry attempts
    
    Returns:
        dict: {
            'success': bool,
            'method_used': str,
            'attempts_made': int,
            'text_sent': str
        }
    """
    if not text or not text.strip():
        return {
            'success': False,
            'method_used': 'none',
            'attempts_made': 0,
            'text_sent': '',
            'error': 'Empty text provided'
        }
    
    # Clean and prepare text
    original_text = text
    # Sanitize newlines for adb input
    original_text = original_text.replace("\r", " ").replace("\n", " ").strip()
    methods = [
        ('adb_shell_simple', lambda t: device.shell(f'input text {t}')),
        ('keyevent_typing', lambda t: _type_with_keyevents(device, t)),
    ]
    
    for attempt in range(max_attempts):
        for method_name, method_func in methods:
            try:
                print(f"📝 Attempt {attempt + 1}/{max_attempts} - Method: {method_name}")
                print(f"📝 Text to input: {original_text[:50]}...")
                
                # Prepare text based on method
                if method_name == 'adb_shell_simple':
                    # Prepare for adb 'input text': spaces as %s, remove shell-breakers
                    prepared_text = original_text
                    prepared_text = " ".join(prepared_text.split())
                    prepared_text = prepared_text.replace(" ", "%s")
                    for ch in ['"', "'", '`', '&', '|', ';', '<', '>', '\\']:
                        prepared_text = prepared_text.replace(ch, '')
                else:
                    prepared_text = original_text
                
                # Execute the method
                method_func(prepared_text)
                time.sleep(1.5)  # Give time for text to appear
                
                print(f"✅ Text input successful with {method_name}")
                return {
                    'success': True,
                    'method_used': method_name,
                    'attempts_made': attempt + 1,
                    'text_sent': original_text
                }
                
            except Exception as e:
                print(f"❌ Method {method_name} failed: {e}")
                time.sleep(0.5)
                continue
    
    # All methods failed
    print(f"❌ All text input methods failed after {max_attempts} attempts")
    return {
        'success': False,
        'method_used': 'failed',
        'attempts_made': max_attempts,
        'text_sent': original_text,
        'error': 'All input methods failed'
    }


def _type_with_keyevents(device, text):
    """Type text using individual key events (slower but more reliable)"""
    for char in text:
        if char == ' ':
            device.shell("input keyevent KEYCODE_SPACE")
        elif char.isalpha():
            # Handle letters
            keycode = f"KEYCODE_{char.upper()}"
            device.shell(f"input keyevent {keycode}")
        elif char.isdigit():
            # Handle numbers
            keycodes = {
                '0': 'KEYCODE_0', '1': 'KEYCODE_1', '2': 'KEYCODE_2',
                '3': 'KEYCODE_3', '4': 'KEYCODE_4', '5': 'KEYCODE_5', 
                '6': 'KEYCODE_6', '7': 'KEYCODE_7', '8': 'KEYCODE_8', '9': 'KEYCODE_9'
            }
            device.shell(f"input keyevent {keycodes[char]}")
        elif char in ".,!?":
            # Handle basic punctuation
            punctuation_codes = {
                '.': 'KEYCODE_PERIOD',
                ',': 'KEYCODE_COMMA', 
                '!': 'KEYCODE_1',  # Shift + 1
                '?': 'KEYCODE_SLASH'  # Shift + /
            }
            if char in ['!', '?']:
                device.shell("input keyevent KEYCODE_SHIFT_LEFT")
            device.shell(f"input keyevent {punctuation_codes[char]}")
        # Skip other special characters
        time.sleep(0.1)  # Small delay between keystrokes


def swipe(device, x1, y1, x2, y2, duration=500):
    device.shell(f"input swipe {x1} {y1} {x2} {y2} {duration}")


def generate_comment(profile_text):
    """Legacy function - now uses OpenAI via analyzer facade"""
    from analyzer import generate_comment
    return generate_comment(profile_text)


def get_screen_resolution(device):
    output = device.shell("wm size")
    print("screen size: ", output)
    resolution = output.strip().split(":")[1].strip()
    width, height = map(int, resolution.split("x"))
    return width, height


def detect_like_button_cv(screenshot_path):
    """
    Detect like button using OpenCV template matching
    
    Returns:
        dict: {
            'found': bool,
            'x': int, 
            'y': int,
            'confidence': float,
            'width': int,
            'height': int
        }
    """
    try:
        # Load template image
        template_path = "assets/like_button.png"
        if not os.path.exists(template_path):
            print(f"❌ Like button template not found: {template_path}")
            return {'found': False, 'confidence': 0.0}
        
        # Load screenshot and template
        screenshot = cv2.imread(screenshot_path)
        template = cv2.imread(template_path)
        
        if screenshot is None:
            print(f"❌ Could not load screenshot: {screenshot_path}")
            return {'found': False, 'confidence': 0.0}
            
        if template is None:
            print(f"❌ Could not load template: {template_path}")
            return {'found': False, 'confidence': 0.0}
        
        # Get template dimensions
        template_height, template_width = template.shape[:2]
        
        # Convert to grayscale for better matching
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # Perform template matching
        result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        
        # Find the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # max_val is the confidence score (0-1)
        confidence = float(max_val)
        
        # Calculate center coordinates
        top_left = max_loc
        center_x = top_left[0] + template_width // 2
        center_y = top_left[1] + template_height // 2
        
        # Consider it found if confidence is above threshold
        confidence_threshold = 0.7
        found = confidence >= confidence_threshold
        
        print(f"🎯 CV Like Button Detection:")
        print(f"   📍 Center: ({center_x}, {center_y})")
        print(f"   📐 Template size: {template_width}x{template_height}")
        print(f"   🎯 Confidence: {confidence:.3f}")
        print(f"   ✅ Found: {found} (threshold: {confidence_threshold})")
        
        return {
            'found': found,
            'x': center_x,
            'y': center_y, 
            'confidence': confidence,
            'width': template_width,
            'height': template_height,
            'top_left_x': top_left[0],
            'top_left_y': top_left[1]
        }
        
    except Exception as e:
        print(f"❌ CV like button detection failed: {e}")
        return {'found': False, 'confidence': 0.0}


def detect_send_button_cv(screenshot_path, preferred: str = None):
    """
    Detect send button using OpenCV template matching.
    Tries both standard and priority send button templates and returns the best match.
    
    Returns:
        dict: {
            'found': bool,
            'x': int, 
            'y': int,
            'confidence': float,
            'width': int,
            'height': int,
            'top_left_x': int,
            'top_left_y': int,
            'matched_template': str
        }
    """
    try:
        # Load screenshot
        screenshot = cv2.imread(screenshot_path)
        if screenshot is None:
            print(f"❌ Could not load screenshot: {screenshot_path}")
            return {'found': False, 'confidence': 0.0}
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Candidate templates: order depends on preferred mode
        standard = ("assets/send_button.png", "send")
        priority = ("assets/send_priority_button.png", "send_priority")
        if preferred == "normal":
            candidates = [standard, priority]
        else:
            candidates = [priority, standard]

        best = {
            'found': False,
            'x': 0,
            'y': 0,
            'confidence': 0.0,
            'width': 0,
            'height': 0,
            'top_left_x': 0,
            'top_left_y': 0,
            'matched_template': ""
        }
        confidence_threshold = 0.6  # single threshold for simplicity

        for template_path, label in candidates:
            if not os.path.exists(template_path):
                print(f"⚠️  Send template not found: {template_path}")
                continue

            template = cv2.imread(template_path)
            if template is None:
                print(f"❌ Could not load template: {template_path}")
                continue

            th, tw = template.shape[:2]
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            # Perform template matching
            result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            confidence = float(max_val)
            top_left = max_loc
            center_x = top_left[0] + tw // 2
            center_y = top_left[1] + th // 2
            found = confidence >= confidence_threshold

            print(f"🎯 CV Send Button Detection ({label}):")
            print(f"   📍 Center: ({center_x}, {center_y})")
            print(f"   📐 Template size: {tw}x{th}")
            print(f"   🎯 Confidence: {confidence:.3f}")
            print(f"   ✅ Found: {found} (threshold: {confidence_threshold})")

            if confidence > best['confidence']:
                best.update({
                    'found': found,
                    'x': center_x,
                    'y': center_y,
                    'confidence': confidence,
                    'width': tw,
                    'height': th,
                    'top_left_x': top_left[0],
                    'top_left_y': top_left[1],
                    'matched_template': label
                })

        if best['matched_template']:
            print(f"✅ Best send match: {best['matched_template']} with confidence {best['confidence']:.3f} (found={best['found']})")
        else:
            print("❌ No send button templates matched.")

        return best if best['matched_template'] else {'found': False, 'confidence': 0.0}

    except Exception as e:
        print(f"❌ CV send button detection failed: {e}")
        return {'found': False, 'confidence': 0.0}


def detect_comment_field_cv(screenshot_path):
    """
    Detect comment field using OpenCV template matching
    
    Returns:
        dict: {
            'found': bool,
            'x': int, 
            'y': int,
            'confidence': float,
            'width': int,
            'height': int
        }
    """
    try:
        # Load template image
        template_path = "assets/comment_field.png"
        if not os.path.exists(template_path):
            print(f"❌ Comment field template not found: {template_path}")
            return {'found': False, 'confidence': 0.0}
        
        # Load screenshot and template
        screenshot = cv2.imread(screenshot_path)
        template = cv2.imread(template_path)
        
        if screenshot is None:
            print(f"❌ Could not load screenshot: {screenshot_path}")
            return {'found': False, 'confidence': 0.0}
            
        if template is None:
            print(f"❌ Could not load template: {template_path}")
            return {'found': False, 'confidence': 0.0}
        
        # Get template dimensions
        template_height, template_width = template.shape[:2]
        
        # Convert to grayscale for better matching
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # Perform template matching
        result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        
        # Find the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # max_val is the confidence score (0-1)
        confidence = float(max_val)
        
        # Calculate center coordinates
        top_left = max_loc
        center_x = top_left[0] + template_width // 2
        center_y = top_left[1] + template_height // 2
        
        # Consider it found if confidence is above threshold
        confidence_threshold = 0.6  # Lower threshold for comment field as text may vary
        found = confidence >= confidence_threshold
        
        print(f"🎯 CV Comment Field Detection:")
        print(f"   📍 Center: ({center_x}, {center_y})")
        print(f"   📐 Template size: {template_width}x{template_height}")
        print(f"   🎯 Confidence: {confidence:.3f}")
        print(f"   ✅ Found: {found} (threshold: {confidence_threshold})")
        
        return {
            'found': found,
            'x': center_x,
            'y': center_y, 
            'confidence': confidence,
            'width': template_width,
            'height': template_height,
            'top_left_x': top_left[0],
            'top_left_y': top_left[1]
        }
        
    except Exception as e:
        print(f"❌ CV comment field detection failed: {e}")
        return {'found': False, 'confidence': 0.0}


def open_hinge(device):
    package_name = "co.match.android.matchhinge"
    device.shell(f"monkey -p {package_name} -c android.intent.category.LAUNCHER 1")
    time.sleep(5)


def reset_hinge_app(device):
    """
    Reset the Hinge app by force closing it, clearing from recent apps, and reopening it.
    This refreshes the app state and can help when the agent gets stuck.
    """
    package_name = "co.match.android.matchhinge"
    
    print("🔄 Resetting Hinge app...")
    
    # Step 1: Force stop the app
    print("🛑 Force stopping Hinge app...")
    device.shell(f"am force-stop {package_name}")
    time.sleep(2)
    
    # Step 2: Kill app from background processes
    print("💀 Killing background processes...")
    device.shell(f"am kill {package_name}")
    time.sleep(1)
    
    # Step 3: Go back to home screen
    device.shell("input keyevent KEYCODE_HOME")
    time.sleep(2)
    
    # Step 4: Reopen the app
    print("🚀 Reopening Hinge app...")
    device.shell(f"am start -n {package_name}")
    time.sleep(2)
    
    print("✅ Hinge app reset completed")
    
    
def detect_keyboard_tick_cv(screenshot_path, template_path="assets/tick.png"):
    """
    Detect keyboard 'tick' button (bottom-right) using OpenCV template matching.
    
    Returns:
        dict: {
            'found': bool,
            'x': int, 
            'y': int,
            'confidence': float,
            'width': int,
            'height': int,
            'top_left_x': int,
            'top_left_y': int
        }
    """
    try:
        # Load screenshot
        screenshot = cv2.imread(screenshot_path)
        if screenshot is None:
            print(f"❌ Could not load screenshot: {screenshot_path}")
            return {'found': False, 'confidence': 0.0}
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Load tick template
        if not os.path.exists(template_path):
            print(f"❌ Tick template not found: {template_path}")
            return {'found': False, 'confidence': 0.0}
        template = cv2.imread(template_path)
        if template is None:
            print(f"❌ Could not load template: {template_path}")
            return {'found': False, 'confidence': 0.0}
        th, tw = template.shape[:2]
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        confidence = float(max_val)
        top_left = max_loc
        center_x = top_left[0] + tw // 2
        center_y = top_left[1] + th // 2
        threshold = 0.5  # slightly lower as tick is small; adjust if noisy
        found = confidence >= threshold

        print(f"🎯 CV Tick Detection:")
        print(f"   📍 Center: ({center_x}, {center_y})")
        print(f"   📐 Template size: {tw}x{th}")
        print(f"   🎯 Confidence: {confidence:.3f}")
        print(f"   ✅ Found: {found} (threshold: {threshold})")

        return {
            'found': found,
            'x': center_x,
            'y': center_y,
            'confidence': confidence,
            'width': tw,
            'height': th,
            'top_left_x': top_left[0],
            'top_left_y': top_left[1]
        }

    except Exception as e:
        print(f"❌ CV tick detection failed: {e}")
        return {'found': False, 'confidence': 0.0}


def detect_age_icon_cv(screenshot_path, template_path="assets/icon_age.png"):
    """
    Detect the profile 'age' icon using OpenCV template matching to infer the Y
    coordinate of the horizontal photo scroller.

    Returns:
        dict: {
            'found': bool,
            'x': int,
            'y': int,
            'confidence': float,
            'width': int,
            'height': int,
            'top_left_x': int,
            'top_left_y': int
        }
    """
    try:
        # Load screenshot
        screenshot = cv2.imread(screenshot_path)
        if screenshot is None:
            print(f"❌ Could not load screenshot: {screenshot_path}")
            return {'found': False, 'confidence': 0.0}
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Load age icon template
        if not os.path.exists(template_path):
            print(f"❌ Age icon template not found: {template_path}")
            return {'found': False, 'confidence': 0.0}
        template = cv2.imread(template_path)
        if template is None:
            print(f"❌ Could not load template: {template_path}")
            return {'found': False, 'confidence': 0.0}
        th, tw = template.shape[:2]
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        confidence = float(max_val)
        top_left = max_loc
        center_x = top_left[0] + tw // 2
        center_y = top_left[1] + th // 2

        # Threshold similar to like/comment buttons, tune as needed
        threshold = 0.65
        found = confidence >= threshold

        print("🎯 CV Age Icon Detection:")
        print(f"   📍 Center: ({center_x}, {center_y})")
        print(f"   📐 Template size: {tw}x{th}")
        print(f"   🎯 Confidence: {confidence:.3f}")
        print(f"   ✅ Found: {found} (threshold: {threshold})")

        return {
            'found': found,
            'x': center_x,
            'y': center_y,
            'confidence': confidence,
            'width': tw,
            'height': th,
            'top_left_x': top_left[0],
            'top_left_y': top_left[1]
        }

    except Exception as e:
        print(f"❌ CV age icon detection failed: {e}")
        return {'found': False, 'confidence': 0.0}


def detect_age_icon_cv_multi(
    screenshot_path: str,
    template_path: Union[str, Sequence[str]] = "assets/icon_age.png",
    roi_top: float = 0.0,
    roi_bottom: float = 0.55,
    scales: list = None,
    threshold: float = 0.55,
    use_edges: bool = True,
    save_debug: bool = True,
    label: str = "age",
    expected_px: int = 60,
    scale_tolerance: float = 0.30,
    min_px: int = 20,
    max_roi_frac: float = 0.12,
    edges_dilate_iter: int = 1
) -> dict:
    """
    Robust age/gender icon detection (multi-template, multi-scale).
    - Accepts a single template path or a list/tuple of template paths.
    - Searches within a vertical ROI (fraction of screen height).
    - Tries multiple scales per template.
    - Optionally matches on edges to reduce tint/contrast sensitivity.
    - If template has alpha channel, uses it as a mask with TM_CCORR_NORMED (non-edges mode).
    - Writes a debug overlay image when save_debug=True for the best match.

    Returns:
        dict: {
            'found': bool,
            'x': int, 'y': int,
            'confidence': float,
            'width': int, 'height': int,
            'top_left_x': int, 'top_left_y': int,
            'scale': float,
            'template_used': str,
            'debug_image_path': str
        }
    """
    try:
        img = cv2.imread(screenshot_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"❌ Could not load screenshot: {screenshot_path}")
            return {'found': False, 'confidence': 0.0}
        h, w = img.shape[:2]
        y0 = max(0, int(h * roi_top))
        y1 = min(h, int(h * roi_bottom))
        roi = img[y0:y1, :].copy()
        roi_h = y1 - y0
        if roi.size == 0:
            print("❌ ROI empty for age icon detection")
            return {'found': False, 'confidence': 0.0}

        # Prepare ROI representations
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if use_edges:
            # Adaptive Canny based on ROI median + dilation to thicken thin strokes
            v = float(np.median(roi_gray))
            lower = int(max(0, (1.0 - 0.33) * v))
            upper = int(min(255, (1.0 + 0.33) * v))
            roi_edges = cv2.Canny(roi_gray, lower, upper)
            kernel = np.ones((3, 3), np.uint8)
            try:
                roi_edges = cv2.dilate(roi_edges, kernel, iterations=int(edges_dilate_iter))
            except Exception:
                pass
        else:
            roi_edges = None

        # Build scale list
        if not scales:
            # Keep placeholder; per-template scale list will be derived around expected_px
            scales = None

        # Normalize template paths to a list
        tpl_paths: List[str] = list(template_path) if isinstance(template_path, (list, tuple)) else [template_path]

        def _load_template_with_alpha(path: str):
            tpl_rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if tpl_rgba is None:
                print(f"❌ Could not load age/gender icon template: {path}")
                return None, None
            if tpl_rgba.ndim == 3 and tpl_rgba.shape[2] == 4:
                bgr = tpl_rgba[:, :, :3]
                alpha = tpl_rgba[:, :, 3]
            else:
                bgr = tpl_rgba if tpl_rgba.ndim == 3 else cv2.cvtColor(tpl_rgba, cv2.COLOR_GRAY2BGR)
                alpha = None
            tpl_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            return tpl_gray, alpha

        best = {
            'found': False, 'confidence': 0.0, 'x': 0, 'y': 0,
            'width': 0, 'height': 0, 'top_left_x': 0, 'top_left_y': 0,
            'scale': 1.0, 'template_used': "", 'debug_image_path': ""
        }
        top_candidates: List[Dict[str, Any]] = []

        for path in tpl_paths:
            tpl_gray, alpha = _load_template_with_alpha(path)
            if tpl_gray is None:
                continue

            # Build per-template scale list around expected pixel height
            if scales:
                scales_local = scales
            elif expected_px and expected_px > 0:
                tg_h = int(tpl_gray.shape[0]) if tpl_gray is not None else 0
                s0 = (float(expected_px) / float(tg_h)) if tg_h > 0 else 0.0
                s_min = max(0.01, s0 * (1.0 - float(scale_tolerance)))
                s_max = max(s_min + 0.005, s0 * (1.0 + float(scale_tolerance)))
                # 15 steps across the band
                scales_local = [float(x) for x in np.linspace(s_min, s_max, num=15)]
            else:
                # Fallback generic sweep if expected_px unavailable
                scales_local = [round(s, 2) for s in np.arange(0.3, 2.51, 0.1).tolist()]

            for s in scales_local:
                tw = max(1, int(tpl_gray.shape[1] * s))
                th = max(1, int(tpl_gray.shape[0] * s))
                if tw <= 1 or th <= 1:
                    continue
                # Size clamps: avoid vanishingly small or overly large kernels relative to ROI height
                max_allowed = int(max(1, roi_h * float(max_roi_frac)))
                if th < int(min_px) or th > max_allowed:
                    continue

                tpl_s = cv2.resize(tpl_gray, (tw, th), interpolation=cv2.INTER_AREA)

                # Choose ROI and method based on edges vs grayscale
                if use_edges:
                    if roi_edges is None or roi_edges.shape[0] < th or roi_edges.shape[1] < tw:
                        continue
                    # Adaptive Canny + dilation for template edges
                    median_tpl = float(np.median(tpl_s))
                    lower_t = int(max(0, (1.0 - 0.33) * median_tpl))
                    upper_t = int(min(255, (1.0 + 0.33) * median_tpl))
                    tpl_proc = cv2.Canny(tpl_s, lower_t, upper_t)
                    try:
                        tpl_proc = cv2.dilate(tpl_proc, kernel, iterations=int(edges_dilate_iter))
                    except Exception:
                        pass
                    if roi_edges.shape[0] < tpl_proc.shape[0] or roi_edges.shape[1] < tpl_proc.shape[1]:
                        continue
                    res = cv2.matchTemplate(roi_edges, tpl_proc, cv2.TM_CCOEFF_NORMED)
                else:
                    roi_proc = roi_gray
                    if roi_proc.shape[0] < th or roi_proc.shape[1] < tw:
                        continue
                    # If template has alpha, use it as mask with TM_CCORR_NORMED
                    if alpha is not None:
                        mask_s = cv2.resize(alpha, (tw, th), interpolation=cv2.INTER_NEAREST)
                        if mask_s.dtype != np.uint8:
                            mask_s = mask_s.astype(np.uint8)
                        res = cv2.matchTemplate(roi_proc, tpl_s, cv2.TM_CCORR_NORMED, mask=mask_s)
                    else:
                        # Build near-white background mask for white PNGs (keep icon strokes only)
                        tpl_gray_s = tpl_s
                        _, mask_s = cv2.threshold(tpl_gray_s, 250, 255, cv2.THRESH_BINARY_INV)
                        mask_s = mask_s.astype(np.uint8)
                        if mask_s.shape != tpl_gray_s.shape:
                            mask_s = cv2.resize(mask_s, (tw, th), interpolation=cv2.INTER_NEAREST)
                        res = cv2.matchTemplate(roi_proc, tpl_s, cv2.TM_CCORR_NORMED, mask=mask_s)

                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                # Compute full-image candidate center for logging/top-k
                top_left = max_loc
                center_x = top_left[0] + tw // 2
                center_y = top_left[1] + th // 2
                full_x = center_x
                full_y = y0 + center_y
                # Track top candidates across templates/scales
                try:
                    top_candidates.append({
                        "confidence": float(max_val),
                        "scale": float(s),
                        "template": os.path.basename(path),
                        "x": int(full_x),
                        "y": int(full_y)
                    })
                except Exception:
                    pass
                # Per-template CV overlay (when gated), shows the top match for this template/scale
                if save_debug and _should_emit_cv_debug():
                    try:
                        dbg_tpl = img.copy()
                        tl_full = (int(max_loc[0]), int(y0 + max_loc[1]))
                        br_full = (tl_full[0] + int(tw), tl_full[1] + int(th))
                        color = (0, 255, 255)  # yellow-ish for per-template overlays
                        cv2.rectangle(dbg_tpl, tl_full, br_full, color, 2)
                        label = f"tpl={os.path.basename(path)} conf={float(max_val):.2f} s={float(s):.2f}"
                        cv2.putText(
                            dbg_tpl,
                            label,
                            (tl_full[0], max(0, tl_full[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                            cv2.LINE_AA,
                        )
                        os.makedirs("images", exist_ok=True)
                        cv_path = os.path.join(
                            "images",
                            f"debug_cv_{os.path.splitext(os.path.basename(path))[0]}_{int(time.time()*1000)}.png",
                        )
                        cv2.imwrite(cv_path, dbg_tpl)
                    except Exception:
                        pass
                if float(max_val) > best['confidence']:
                    top_left = max_loc
                    center_x = top_left[0] + tw // 2
                    center_y = top_left[1] + th // 2
                    full_x = center_x
                    full_y = y0 + center_y
                    best.update({
                        'confidence': float(max_val),
                        'x': int(full_x),
                        'y': int(full_y),
                        'width': int(tw),
                        'height': int(th),
                        'top_left_x': int(top_left[0]),
                        'top_left_y': int(y0 + top_left[1]),
                        'scale': float(s),
                        'template_used': path,
                    })

        best['found'] = best['confidence'] >= threshold

        # Debug overlay for the best match
        if save_debug:
            dbg = img.copy()
            # Draw ROI bounds for clarity
            cv2.line(dbg, (0, y0), (w, y0), (255, 255, 0), 1)
            cv2.line(dbg, (0, y1), (w, y1), (255, 255, 0), 1)
            cv2.putText(
                dbg,
                f"ROI {roi_top:.2f}-{roi_bottom:.2f}",
                (10, max(20, y0 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if best['width'] > 0 and best['height'] > 0:
                tl = (best['top_left_x'], best['top_left_y'])
                br = (best['top_left_x'] + best['width'], best['top_left_y'] + best['height'])
                color = (0, 255, 0) if best['found'] else (0, 0, 255)
                cv2.rectangle(dbg, tl, br, color, 3)
                label_text = f"conf={best['confidence']:.2f}, s={best['scale']:.2f}"
                if best.get('template_used'):
                    label_text += f", tpl={os.path.basename(best['template_used'])}"
                cv2.putText(
                    dbg, label_text,
                    (tl[0], max(0, tl[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
                )
                # Crosshair at candidate center
                try:
                    cx, cy = int(best.get('x', 0)), int(best.get('y', 0))
                    cv2.drawMarker(dbg, (cx, cy), color, markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
                except Exception:
                    pass
            kind = "gender" if (label and label.lower() == "gender") else "age"
            dbg_path = os.path.join("images", f"debug_{kind}_icon_{int(time.time()*1000)}.png")
            try:
                os.makedirs("images", exist_ok=True)
                cv2.imwrite(dbg_path, dbg)
                best['debug_image_path'] = dbg_path
                print(f"🖼️  {kind.title()} icon debug overlay: {dbg_path}")
                # Save top-3 candidates JSON alongside the overlay for quick inspection
                try:
                    top3 = sorted(top_candidates, key=lambda c: c.get("confidence", 0.0), reverse=True)[:3]
                    cand_path = os.path.join("images", f"debug_cv_candidates_{kind}_{int(time.time()*1000)}.json")
                    with open(cand_path, "w", encoding="utf-8") as jf:
                        json.dump(top3, jf, ensure_ascii=False, indent=2)
                    best['candidates_path'] = cand_path
                except Exception:
                    pass
            except Exception as _:
                pass

        print(f"🎯 {label.title()} icon multi-scale result: found={best['found']} conf={best['confidence']:.3f} scale={best['scale']:.2f} tpl={os.path.basename(best.get('template_used',''))}")
        print(f"[CV] {label} ROI=({roi_top:.2f},{roi_bottom:.2f}) y0={y0} y1={y1} cov={(roi_bottom - roi_top):.2f} x={best.get('x',0)} y={best.get('y',0)}")
        return best

    except Exception as e:
        print(f"❌ Multi-scale age icon detection failed: {e}")
        return {'found': False, 'confidence': 0.0}


def detect_age_row_dual_templates(
    screenshot_path: str,
    template_paths: Sequence[str],
    roi_top: float = 0.0,
    roi_bottom: float = 0.55,
    threshold: float = 0.55,
    use_edges: bool = True,
    save_debug: bool = True,
    tolerance_px: int = 5,
    tolerance_ratio: float = 0.005,
    require_both: bool = True,
    expected_px: int = 60,
    scale_tolerance: float = 0.30,
    min_px: int = 20,
    max_roi_frac: float = 0.12,
    edges_dilate_iter: int = 1
) -> dict:
    """
    Run detection for two templates (e.g., age and gender icons) and derive a consensus Y.
    - Both icons should lie on the same horizontal row; if their Y differ slightly, take the average.
    - If their Y differ more than tolerance, return found=False with error="y_mismatch".
    - If one is missing and require_both is True, return found=False with error="one_icon_missing".
    - Writes a composite overlay showing both boxes and lines (age=green, gender=blue, avg=magenta).
    """
    try:
        tpls = list(template_paths or [])
        # New: if 3+ templates are provided, perform 2-of-3 consensus (age/gender/height)
        if len(tpls) >= 3:
            try:
                img = cv2.imread(screenshot_path, cv2.IMREAD_COLOR)
                if img is None:
                    return {'found': False, 'error': 'load_image_failed'}
                h, w = img.shape[:2]
                tol = max(int(tolerance_px), int(h * float(tolerance_ratio)))
                # Detect each template independently
                labels = []
                for p in tpls:
                    name = os.path.basename(p).lower()
                    if 'height' in name:
                        labels.append('height')
                    elif 'gender' in name:
                        labels.append('gender')
                    else:
                        labels.append('age')
                results = []
                for i, p in enumerate(tpls):
                    lbl = labels[i] if i < len(labels) else 'age'
                    res = detect_age_icon_cv_multi(
                        screenshot_path,
                        template_path=p,
                        roi_top=roi_top,
                        roi_bottom=roi_bottom,
                        threshold=threshold,
                        use_edges=use_edges,
                        save_debug=False,
                        label=lbl,
                        expected_px=expected_px,
                        scale_tolerance=scale_tolerance,
                        min_px=min_px,
                        max_roi_frac=max_roi_frac,
                        edges_dilate_iter=edges_dilate_iter
                    )
                    res['template'] = p
                    res['label'] = lbl
                    results.append(res)
                # Filter valid detections
                valids = [r for r in results if r.get('found')]
                # Build best 2-of-3 (or 3/3) consensus
                y_avg = None
                chosen = []
                if len(valids) >= 2:
                    # Evaluate all pairs within tolerance and choose highest combined confidence
                    best_pair = None
                    best_score = -1.0
                    for i in range(len(valids)):
                        for j in range(i+1, len(valids)):
                            y1 = int(valids[i].get('y', 0))
                            y2 = int(valids[j].get('y', 0))
                            dy = abs(y1 - y2)
                            if dy <= tol:
                                score = float(valids[i].get('confidence', 0.0)) + float(valids[j].get('confidence', 0.0))
                                if score > best_score:
                                    best_score = score
                                    best_pair = (valids[i], valids[j])
                    if best_pair:
                        y1 = int(best_pair[0].get('y', 0))
                        y2 = int(best_pair[1].get('y', 0))
                        y_avg = int(round((y1 + y2) / 2))
                        chosen = [best_pair[0], best_pair[1]]
                        # If a third valid within tolerance of y_avg, include it in average
                        for r in valids:
                            if r in chosen:
                                continue
                            if abs(int(r.get('y', 0)) - y_avg) <= tol:
                                chosen.append(r)
                                y_avg = int(round((y_avg * (len(chosen)-1) + int(r.get('y',0))) / len(chosen)))
                found = y_avg is not None
                # Build composite overlay
                dbg = img.copy()
                colors = {
                    'age': (0, 255, 0),
                    'gender': (255, 0, 0),
                    'height': (0, 165, 255)
                }
                for r in results:
                    if r.get('width', 0) > 0 and r.get('height', 0) > 0:
                        tl = (int(r.get('top_left_x', 0)), int(r.get('top_left_y', 0)))
                        br = (tl[0] + int(r.get('width', 0)), tl[1] + int(r.get('height', 0)))
                        col = colors.get(r.get('label', 'age'), (200, 200, 200))
                        cv2.rectangle(dbg, tl, br, col, 3)
                        try:
                            cv2.line(dbg, (0, int(r.get('y', 0))), (w, int(r.get('y', 0))), col, 2)
                        except Exception:
                            pass
                        try:
                            cv2.putText(dbg, f"{os.path.basename(r.get('template',''))} conf={float(r.get('confidence',0.0)):.2f}",
                                        (tl[0], max(0, tl[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
                        except Exception:
                            pass
                if y_avg is not None:
                    cv2.line(dbg, (0, y_avg), (w, y_avg), (255, 0, 255), 2)
                label = f"found={found} min2of3 tol={tol}"
                cv2.putText(dbg, label, (10, max(30, int(h*0.03))), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2, cv2.LINE_AA)
                dbg_path = os.path.join("images", f"debug_age_two_of_three_{int(time.time()*1000)}.png")
                try:
                    os.makedirs("images", exist_ok=True)
                    cv2.imwrite(dbg_path, dbg)
                except Exception:
                    dbg_path = ""
                return {
                    'found': bool(found),
                    'y': int(y_avg) if y_avg is not None else 0,
                    'method': 'two_of_three',
                    'results': [
                        {
                            'template': r.get('template'),
                            'found': bool(r.get('found')),
                            'y': int(r.get('y', 0)),
                            'confidence': float(r.get('confidence', 0.0)),
                            'width': int(r.get('width', 0)),
                            'height': int(r.get('height', 0)),
                        } for r in results
                    ],
                    'delta_y': None,
                    'tolerance': int(tol),
                    'debug_image_path': dbg_path,
                    'error': None if found else 'insufficient_matches'
                }
            except Exception as _e:
                print(f"❌ Two-of-three age row detection failed: {_e}")
                return {'found': False, 'error': 'exception'}
        if len(tpls) < 2:
            single = detect_age_icon_cv_multi(
                screenshot_path,
                template_path=tpls[0] if tpls else "assets/icon_age.png",
                roi_top=roi_top,
                roi_bottom=roi_bottom,
                threshold=threshold,
                use_edges=use_edges,
                save_debug=save_debug,
                label="age",
                expected_px=expected_px,
                scale_tolerance=scale_tolerance,
                min_px=min_px,
                max_roi_frac=max_roi_frac,
                edges_dilate_iter=edges_dilate_iter
            )
            single['method'] = 'single_template'
            single['results'] = [single]
            single['delta_y'] = 0
            single['tolerance'] = 0
            single['error'] = None if single.get('found') else 'single_template_only'
            return single

        res1 = detect_age_icon_cv_multi(
            screenshot_path,
            template_path=tpls[0],
            roi_top=roi_top,
            roi_bottom=roi_bottom,
            threshold=threshold,
            use_edges=use_edges,
            save_debug=save_debug,
            label="age",
            expected_px=expected_px,
            scale_tolerance=scale_tolerance,
            min_px=min_px,
            max_roi_frac=max_roi_frac,
            edges_dilate_iter=edges_dilate_iter
        )
        res2 = detect_age_icon_cv_multi(
            screenshot_path,
            template_path=tpls[1],
            roi_top=roi_top,
            roi_bottom=roi_bottom,
            threshold=threshold,
            use_edges=use_edges,
            save_debug=save_debug,
            label="gender",
            expected_px=expected_px,
            scale_tolerance=scale_tolerance,
            min_px=min_px,
            max_roi_frac=max_roi_frac,
            edges_dilate_iter=edges_dilate_iter
        )

        img = cv2.imread(screenshot_path, cv2.IMREAD_COLOR)
        if img is None:
            return {'found': False, 'error': 'load_image_failed'}

        h, w = img.shape[:2]
        tol = max(int(tolerance_px), int(h * float(tolerance_ratio)))

        both_found = bool(res1.get('found')) and bool(res2.get('found'))
        error = None
        y_avg = None
        delta_y = None

        if both_found:
            y1 = int(res1.get('y', 0))
            y2 = int(res2.get('y', 0))
            delta_y = abs(y1 - y2)
            if delta_y <= tol:
                y_avg = int(round((y1 + y2) / 2))
            else:
                error = "y_mismatch"
        else:
            if require_both:
                error = "one_icon_missing"

        found = (y_avg is not None)

        # Composite overlay
        dbg = img.copy()
        if res1.get('width', 0) > 0 and res1.get('height', 0) > 0:
            tl1 = (int(res1.get('top_left_x', 0)), int(res1.get('top_left_y', 0)))
            br1 = (tl1[0] + int(res1.get('width', 0)), tl1[1] + int(res1.get('height', 0)))
            cv2.rectangle(dbg, tl1, br1, (0, 255, 0), 3)
            cv2.putText(dbg, f"{os.path.basename(res1.get('template_used',''))} conf={res1.get('confidence',0.0):.2f}",
                        (tl1[0], max(0, tl1[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
        if res2.get('width', 0) > 0 and res2.get('height', 0) > 0:
            tl2 = (int(res2.get('top_left_x', 0)), int(res2.get('top_left_y', 0)))
            br2 = (tl2[0] + int(res2.get('width', 0)), tl2[1] + int(res2.get('height', 0)))
            cv2.rectangle(dbg, tl2, br2, (255, 0, 0), 3)
            cv2.putText(dbg, f"{os.path.basename(res2.get('template_used',''))} conf={res2.get('confidence',0.0):.2f}",
                        (tl2[0], max(0, tl2[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv2.LINE_AA)

        if res1.get('found'):
            y1 = int(res1.get('y', 0))
            cv2.line(dbg, (0, y1), (w, y1), (0, 255, 0), 2)
        if res2.get('found'):
            y2 = int(res2.get('y', 0))
            cv2.line(dbg, (0, y2), (w, y2), (255, 0, 0), 2)
        if y_avg is not None:
            cv2.line(dbg, (0, y_avg), (w, y_avg), (255, 0, 255), 2)

        label = f"found={found}"
        if delta_y is not None:
            label += f" delta_y={delta_y}"
        label += f" tol={tol}"
        if error:
            label += f" err={error}"
        cv2.putText(dbg, label, (10, max(30, int(h*0.03))), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2, cv2.LINE_AA)

        dbg_path = os.path.join("images", f"debug_age_dual_{int(time.time()*1000)}.png")
        try:
            os.makedirs("images", exist_ok=True)
            cv2.imwrite(dbg_path, dbg)
            print(f"🖼️  Dual-icon debug overlay: {dbg_path}")
        except Exception:
            dbg_path = ""

        return {
            'found': found,
            'y': int(y_avg) if y_avg is not None else 0,
            'method': 'dual_templates',
            'results': [
                {
                    'template': tpls[0],
                    'found': bool(res1.get('found')),
                    'y': int(res1.get('y', 0)),
                    'confidence': float(res1.get('confidence', 0.0)),
                    'width': int(res1.get('width', 0)),
                    'height': int(res1.get('height', 0)),
                },
                {
                    'template': tpls[1],
                    'found': bool(res2.get('found')),
                    'y': int(res2.get('y', 0)),
                    'confidence': float(res2.get('confidence', 0.0)),
                    'width': int(res2.get('width', 0)),
                    'height': int(res2.get('height', 0)),
                }
            ],
            'delta_y': int(delta_y) if delta_y is not None else None,
            'tolerance': int(tol),
            'debug_image_path': dbg_path,
            'error': error
        }
    except Exception as e:
        print(f"❌ Dual-template age row detection failed: {e}")
        return {'found': False, 'error': 'exception'}

def infer_carousel_y_by_edges(
    screenshot_path: str,
    roi_top: float = 0.15,
    roi_bottom: float = 0.60,
    smooth_kernel: int = 21
) -> dict:
    """
    Infer a good horizontal swipe Y by finding the row with strongest vertical edges
    in a top-half ROI (images carousel tends to have strong vertical edges).
    Returns:
        {'found': bool, 'y': int, 'score': float, 'debug_image_path': str}
    """
    try:
        img = cv2.imread(screenshot_path)
        if img is None:
            print(f"❌ Could not load screenshot: {screenshot_path}")
            return {'found': False}
        h, w = img.shape[:2]
        y0 = max(0, int(h * roi_top))
        y1 = min(h, int(h * roi_bottom))
        roi = img[y0:y1, :].copy()
        if roi.size == 0:
            return {'found': False}

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Emphasize vertical edges (Sobel X)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        absx = np.abs(sobelx)
        # Sum of vertical edges per row
        row_strength = absx.sum(axis=1)
        # Smooth to reduce noise
        if smooth_kernel > 1 and smooth_kernel % 2 == 1:
            row_strength = cv2.GaussianBlur(row_strength.reshape(-1,1), (1, smooth_kernel), 0).flatten()
        idx = int(np.argmax(row_strength))
        score = float(row_strength[idx])
        y_guess = y0 + idx

        # Debug overlay: draw horizontal line
        dbg = img.copy()
        cv2.line(dbg, (0, y_guess), (w, y_guess), (255, 0, 0), 2)
        dbg_path = os.path.join("images", f"debug_carousel_y_{int(time.time()*1000)}.png")
        try:
            os.makedirs("images", exist_ok=True)
            cv2.imwrite(dbg_path, dbg)
            print(f"🖼️  Carousel Y debug overlay: {dbg_path} (y={y_guess}, score={score:.1f})")
        except Exception:
            pass

        return {'found': True, 'y': int(y_guess), 'score': score, 'debug_image_path': dbg_path}
    except Exception as e:
        print(f"❌ Failed to infer carousel Y by edges: {e}")
        return {'found': False}


# --- Image similarity (aHash) utilities for dedup/stabilization ---

def perceptual_hash_ahash(image_path: str, hash_size: int = 8) -> np.ndarray:
    """
    Compute a simple average hash (aHash) of an image.
    Returns a boolean numpy array of shape (hash_size*hash_size,) representing bits.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.array([], dtype=np.bool_)
        # Resize to hash_size x hash_size
        small = cv2.resize(img, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
        avg = small.mean()
        bits = (small.flatten() > avg)
        return bits
    except Exception:
        return np.array([], dtype=np.bool_)


def hamming_distance(h1: np.ndarray, h2: np.ndarray) -> int:
    """
    Compute Hamming distance between two boolean bit arrays of same length.
    """
    if h1.size == 0 or h2.size == 0 or h1.shape != h2.shape:
        return 64  # effectively 'very different' for default 8x8
    return int(np.count_nonzero(h1 ^ h2))


def are_images_similar(path1: str, path2: str, hash_size: int = 8, threshold: int = 5) -> bool:
    """
    Compare two images using aHash and return True if they are similar within threshold.
    Lower threshold => stricter equality. Default threshold 5 works well for near-identical frames.
    """
    h1 = perceptual_hash_ahash(path1, hash_size=hash_size)
    h2 = perceptual_hash_ahash(path2, hash_size=hash_size)
    dist = hamming_distance(h1, h2)
    try:
        print(f"🧮 aHash compare: dist={dist} (threshold={threshold}) for:\n  {os.path.basename(path1)}\n  {os.path.basename(path2)}")
    except Exception:
        pass
    return dist <= threshold


def are_images_similar_roi(
    path1: str,
    path2: str,
    y_center: int,
    band_ratio: float = 0.12,
    band_px: int | None = None,
    hash_size: int = 8,
    threshold: int = 5
) -> bool:
    """
    Compare two images using aHash on a horizontal ROI band centered at y_center.
    - band_ratio: fraction of image height to use if band_px not provided.
    - Lower threshold => stricter equality.
    """
    try:
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            # Fall back to full-frame compare if load failed
            return are_images_similar(path1, path2, hash_size=hash_size, threshold=threshold)

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        if h1 <= 0 or h2 <= 0:
            return are_images_similar(path1, path2, hash_size=hash_size, threshold=threshold)

        # Compute band height
        if band_px is None or band_px <= 0:
            band_px = max(8, int(round(min(h1, h2) * float(band_ratio if band_ratio is not None else 0.12))))

        def _crop_band(gray, yc, band):
            hh = gray.shape[0]
            y0 = max(0, int(yc - band // 2))
            y1 = min(hh, y0 + band)
            if y1 - y0 < 4:
                # Fallback: center band of minimum height
                y0 = max(0, min(hh - 4, int(hh * 0.5) - 2))
                y1 = min(hh, y0 + 4)
            return gray[y0:y1, :], (y0, y1)

        roi1, (y0_1, y1_1) = _crop_band(img1, y_center, band_px)
        roi2, (y0_2, y1_2) = _crop_band(img2, y_center, band_px)

        # If ROI invalid/small, fall back to full
        if roi1.size == 0 or roi2.size == 0 or roi1.shape[0] < 4 or roi2.shape[0] < 4:
            return are_images_similar(path1, path2, hash_size=hash_size, threshold=threshold)

        # aHash over ROI
        small1 = cv2.resize(roi1, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
        small2 = cv2.resize(roi2, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
        avg1 = small1.mean()
        avg2 = small2.mean()
        hbits1 = (small1.flatten() > avg1)
        hbits2 = (small2.flatten() > avg2)
        dist = hamming_distance(hbits1, hbits2)

        try:
            print(
                f"🧮 aHash ROI compare: y={y_center} band_px={band_px} "
                f"dist={dist} (threshold={threshold}) for:\n  {os.path.basename(path1)}[{y0_1}:{y1_1}]\n  {os.path.basename(path2)}[{y0_2}:{y1_2}]"
            )
        except Exception:
            pass

        return dist <= threshold
    except Exception:
        # On any error, fall back to full-frame compare
        return are_images_similar(path1, path2, hash_size=hash_size, threshold=threshold)
