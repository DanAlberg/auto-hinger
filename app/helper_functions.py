from ppadb.client import Client as AdbClient
import time
import cv2
import numpy as np
import os
import glob



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


def detect_age_icon_cv(screenshot_path, template_path="assets/age_icon.png"):
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
    template_path: str = "assets/age_icon.png",
    roi_top: float = 0.0,
    roi_bottom: float = 0.55,
    scales: list = None,
    threshold: float = 0.55,
    use_edges: bool = True,
    save_debug: bool = True
) -> dict:
    """
    Robust age icon detection:
    - Searches within a vertical ROI (fraction of screen height).
    - Tries multiple scales of the template.
    - Optionally matches on edges to reduce tint/contrast sensitivity.
    - Writes a debug overlay image when save_debug=True.

    Returns:
        dict: {
            'found': bool,
            'x': int, 'y': int,
            'confidence': float,
            'width': int, 'height': int,
            'top_left_x': int, 'top_left_y': int,
            'scale': float,
            'debug_image_path': str
        }
    """
    try:
        img = cv2.imread(screenshot_path)
        if img is None:
            print(f"❌ Could not load screenshot: {screenshot_path}")
            return {'found': False, 'confidence': 0.0}
        h, w = img.shape[:2]
        y0 = max(0, int(h * roi_top))
        y1 = min(h, int(h * roi_bottom))
        roi = img[y0:y1, :].copy()
        if roi.size == 0:
            print("❌ ROI empty for age icon detection")
            return {'found': False, 'confidence': 0.0}

        tpl = cv2.imread(template_path)
        if tpl is None:
            print(f"❌ Could not load age icon template: {template_path}")
            return {'found': False, 'confidence': 0.0}

        # Prepare base images
        if use_edges:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_proc = cv2.Canny(roi_gray, 50, 150)
        else:
            roi_proc = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        base_tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

        # Build scale list
        if not scales:
            scales = [round(s, 2) for s in np.arange(0.6, 1.51, 0.1).tolist()]

        best = {
            'found': False, 'confidence': 0.0, 'x': 0, 'y': 0,
            'width': 0, 'height': 0, 'top_left_x': 0, 'top_left_y': 0,
            'scale': 1.0, 'debug_image_path': ""
        }

        method = cv2.TM_CCOEFF_NORMED
        for s in scales:
            # Resize template for this scale
            tw = max(1, int(base_tpl_gray.shape[1] * s))
            th = max(1, int(base_tpl_gray.shape[0] * s))
            tpl_s = cv2.resize(base_tpl_gray, (tw, th), interpolation=cv2.INTER_AREA)
            if use_edges:
                tpl_s = cv2.Canny(tpl_s, 50, 150)

            if roi_proc.shape[0] < tpl_s.shape[0] or roi_proc.shape[1] < tpl_s.shape[1]:
                continue

            res = cv2.matchTemplate(roi_proc, tpl_s, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > best['confidence']:
                top_left = max_loc
                center_x = top_left[0] + tw // 2
                center_y = top_left[1] + th // 2
                # Convert to full-image coordinates
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
                })

        best['found'] = best['confidence'] >= threshold

        # Debug overlay
        if save_debug:
            dbg = img.copy()
            if best['width'] > 0 and best['height'] > 0:
                tl = (best['top_left_x'], best['top_left_y'])
                br = (best['top_left_x'] + best['width'], best['top_left_y'] + best['height'])
                color = (0, 255, 0) if best['found'] else (0, 0, 255)
                cv2.rectangle(dbg, tl, br, color, 3)
                cv2.putText(
                    dbg, f"conf={best['confidence']:.2f}, s={best['scale']:.2f}",
                    (tl[0], max(0, tl[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
                )
            dbg_path = os.path.join("images", f"debug_age_icon_{int(time.time()*1000)}.png")
            try:
                os.makedirs("images", exist_ok=True)
                cv2.imwrite(dbg_path, dbg)
                best['debug_image_path'] = dbg_path
                print(f"🖼️  Age icon debug overlay: {dbg_path}")
            except Exception as _:
                pass

        print(f"🎯 Age icon multi-scale result: found={best['found']} conf={best['confidence']:.3f} scale={best['scale']:.2f}")
        return best

    except Exception as e:
        print(f"❌ Multi-scale age icon detection failed: {e}")
        return {'found': False, 'confidence': 0.0}


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
