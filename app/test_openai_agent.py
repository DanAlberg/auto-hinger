#!/usr/bin/env python3
"""
LLM-based test script for Hinge automation system.
Mirrors the previous test flow and uses the analyzer facade.
"""

import os
import config  # ensure .env is loaded early via config.py
from helper_functions import connect_device, get_screen_resolution, capture_screenshot
import analyzer  # facade that re-exports analyzer functions
from llm_client import get_llm_client, get_llm_provider, resolve_model


def test_llm_connection():
    """Test if the configured LLM API connection works properly."""
    provider = get_llm_provider()
    print("Testing LLM API connection...")

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("OPENAI_API_KEY not found in environment variables")
            print("Make sure to set your API key in the .env file:")
            print("   echo 'OPENAI_API_KEY=your-key-here' > .env")
            return False
    elif provider == "gemini":
        use_vertex = os.getenv("GEMINI_USE_VERTEX", "1").strip().lower() in ("1", "true", "yes", "y", "on")
        if use_vertex:
            project_id = os.getenv("GEMINI_PROJECT_ID")
            location = os.getenv("GEMINI_LOCATION")
            if not project_id or not location:
                print("GEMINI_PROJECT_ID or GEMINI_LOCATION missing for Vertex Gemini")
                print("Add GEMINI_PROJECT_ID and GEMINI_LOCATION to .env")
                return False
        else:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("GEMINI_API_KEY not found in environment variables")
                print("Make sure to set your API key in the .env file:")
                print("   echo 'GEMINI_API_KEY=your-key-here' > .env")
                return False
    else:
        print(f"Unknown LLM provider: {provider}")
        return False

    try:
        client = get_llm_client()

        # Simple text generation test
        model = resolve_model("gpt-5-mini")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": 'Say "Hello, LLM test successful!" if you can read this.'}],
            temperature=0.0,
        )

        text = (response.choices[0].message.content or "").strip()
        if text and "successful" in text.lower():
            print("LLM API connection working properly")
            print(f"Response: {text}")
            return True
        print("LLM API responded but response seems unexpected")
        print(f"Response: {text!r}")
        return False

    except Exception as e:
        print(f"LLM API connection failed: {e}")
        return False


def test_agent_config():
    """Test agent configuration loading."""
    print("\nTesting agent configuration...")

    try:
        from agent_config import DEFAULT_CONFIG, FAST_CONFIG

        configs = {
            "default": DEFAULT_CONFIG,
            "fast": FAST_CONFIG,
        }

        for name, config in configs.items():
            print(f"{name.capitalize()} config loaded:")
            print(f"  Quality threshold: {config.quality_threshold_medium}")
            print(f"  Max retries: {config.max_retries_per_action}")

        return True

    except Exception as e:
        print(f"Agent configuration test failed: {e}")
        return False


def test_device_connection():
    """Test ADB device connection."""
    print("\nTesting device connection...")

    try:
        device = connect_device()
        if not device:
            print("No device connected")
            print("Make sure to:")
            print("  1. Enable USB debugging on your Android device")
            print("  2. Connect device via USB")
            print("  3. Authorize computer on device when prompted")
            print("  4. Run 'adb devices' to verify connection")
            return False, None

        print(f"Device connected: {device.serial}")

        # Test screen resolution
        width, height = get_screen_resolution(device)
        print(f"Screen resolution: {width}x{height}")

        return True, device

    except Exception as e:
        print(f"Device connection failed: {e}")
        return False, None


def test_screenshot_and_analysis(device):
    """Test screenshot capture and LLM-based analysis."""
    print("\nTesting screenshot and analysis...")

    if not device:
        print("Skipping screenshot test - no device connected")
        return False

    try:
        # Capture test screenshot
        print("Capturing test screenshot...")
        screenshot_path = capture_screenshot(device, "llm_test")

        if not os.path.exists(screenshot_path):
            print(f"Screenshot not saved to {screenshot_path}")
            return False

        print(f"Screenshot saved: {screenshot_path}")

        # Test text extraction
        print("Testing text extraction with analyzer...")
        extracted_text = analyzer.extract_text_from_image(screenshot_path)

        if extracted_text:
            print("Text extraction successful")
            preview = extracted_text[:120]
            suffix = "..." if len(extracted_text) > 120 else ""
            print(f"Extracted text preview: {preview}{suffix}")
        else:
            print("No text extracted (may be normal if screen has no text)")

        # Test UI analysis
        print("Testing UI analysis with analyzer...")
        ui_analysis = analyzer.analyze_dating_ui(screenshot_path)

        if ui_analysis:
            print("UI analysis successful")
            print(f"Analysis keys: {list(ui_analysis.keys())}")
            if "profile_quality_score" in ui_analysis:
                print(f"Profile quality score: {ui_analysis.get('profile_quality_score', 'N/A')}")
        else:
            print("UI analysis returned empty (may indicate issue)")

        return True

    except Exception as e:
        print(f"Screenshot and analysis test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Starting LLM-based Tests")
    print("=" * 50)

    results = {
        "llm": test_llm_connection(),
        "config": test_agent_config(),
    }

    device_connected, device = test_device_connection()
    results["device"] = device_connected
    results["screenshot"] = test_screenshot_and_analysis(device) if device_connected else False

    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name.capitalize():12} {status}")

    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result is True)

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("All tests passed! System is ready for automation.")
        print("You can now run: uv run python main_agent.py")
    else:
        print("Some tests failed. Please check the issues above.")
        if not results.get("llm"):
            print("Fix LLM API setup first")
        if not results.get("device"):
            print("Fix device connection next")

    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
