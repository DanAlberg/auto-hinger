# Hinge Automation (Fork)

Automates interactions in the Hinge Android app using ADB, OpenCV template matching, and an AI model (OpenAI by default) for screen/text analysis and decision making.

Warning
- This can send automated likes/comments. Review platform terms of service and use at your own risk.
- Screenshots may be analyzed by an external AI API if enabled. Do not run with private data you’re unwilling to share.
- Start with a test account and conservative settings.

## Features
- AI-driven workflow (LangGraph) to decide next actions based on current screen/state
- OpenCV template matching for UI elements (like button, comment field, send button)
- ADB-based device control (taps, swipes, typing, app reset)
- Profile text extraction and analysis from screenshots (OpenAI vision)
- Profile-change verification (text/feature comparison) for reliable action success checks
- Optional personalized comment generation and sending
- Error handling, recovery attempts, and batch processing

## Tech Stack
- Python 3.13+ (dependency management via uv)
- OpenAI (default model: gpt-4o-mini) for vision + reasoning
- LangGraph for stateful agent orchestration
- OpenCV for computer vision (template matching)
- ADB (Android Debug Bridge) for device automation
- Dockerfile included for containerization (optional)

## Requirements
- Android device with USB debugging enabled and Hinge installed
- Android SDK Platform Tools (adb)
- Python 3.13+
- uv package manager (recommended)
- OpenAI API key (default). Gemini support would require swapping analyzer components.

## Setup
- Create environment file at app/.env (do NOT commit this):
  OPENAI_API_KEY=your-openai-api-key
  # Optional if you switch implementations:
  GEMINI_API_KEY=your-gemini-api-key

- Install dependencies (from the app/ directory):
  uv sync

- Verify device:
  adb devices   # should list your device

## Quick Start
- Minimal, safer run (1 profile, verbose, no screenshots saved to disk):
  cd app
  uv run python main_agent.py --profiles 1 --config conservative --verbose --no-screenshots

- Standard run (default config, saves screenshots to app/images/):
  uv run python main_agent.py

## Command Line Options
- --profiles, -p: Maximum number of profiles to process (default: 10)
- --config, -c: Configuration preset: default | fast | conservative (default: default)
- --device-ip: ADB server host (default: 127.0.0.1)
- --verbose, -v: Enable verbose logging
- --no-screenshots: Disable saving screenshots to disk

Examples
- Process 20 profiles with verbose logging:
  uv run python main_agent.py --profiles 20 --verbose

- Use fast preset for more throughput (less thorough):
  uv run python main_agent.py --config fast --profiles 5

- Conservative preset for safer automation:
  uv run python main_agent.py --config conservative --profiles 3

## Architecture (Overview)
- Entry point: app/main_agent.py
  - Parses CLI, applies AgentConfig, runs LangGraphHingeAgent.
- Agent core: app/langgraph_hinge_agent.py
  - Nodes include initialize_session, ai_decide_action, capture_screenshot, analyze_profile, make_like_decision, detect_like_button, execute_like, generate_comment, send_comment_with_typing, send_like_without_comment, execute_dislike, navigate_to_next, verify_profile_change, recover_from_stuck, reset_app, finalize_session.
  - Uses OpenAI for vision/text analysis and decision JSON; OpenCV for template matching; ADB for interactions.
- CV helpers & ADB: app/helper_functions.py
  - detect_like_button_cv, detect_send_button_cv, detect_comment_field_cv; tap/swipe/keyboard helpers; app reset/open.
- Analyzer facade: app/analyzer.py -> app/analyzer_openai.py (default)
  - Vision OCR, UI analysis, comment generation, outcome verification; can be swapped to a Gemini-based analyzer with changes.
- Data store: app/data_store.py
  - generated_comments.json, and success-rate calculation if feedback_records.json exists.
- Templates: app/assets/
  - like_button.png, comment_field.png, send_button.png (update with device-specific crops if matching is unreliable).
- Images (screenshots): app/images/ (created at runtime).

## Troubleshooting
- Device connection
  adb devices
  adb kill-server && adb start-server

- Dependencies
  cd app && uv sync --reinstall
  cd app && uv sync --frozen

- Verbose logs
  cd app && uv run python main_agent.py --verbose

Notes
- Template matching is sensitive to device/DPI/theme. Replace PNGs in app/assets/ with crops from your device if detection is weak, or adjust confidence thresholds in helper_functions.py.
- Sensitive data: keep app/.env out of version control. See .gitignore.

## Disclaimer
For educational and research purposes only. Use responsibly. The authors are not responsible for misuse or violations of terms of service.

## License
MIT (or match upstream license if different).
