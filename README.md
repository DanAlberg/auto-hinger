# Hinge Automation

Automates interactions in the Hinge Android app using ADB, OpenCV template matching, and an AI model (OpenAI by default) for screen/text analysis and decision making.

Warning
- This can send automated likes/comments. Review platform terms of service and use at your own risk.
- Screenshots may be analyzed by an external AI API if enabled. Do not run with private data you’re unwilling to share.
- Start with a test account and conservative settings.

What this tool does
- Navigates Hinge profiles on a connected Android device (ADB)
- Extracts profile content with vision/analysis
- Makes a deterministic like-with-comment by default (configurable)
- Verifies navigation and logs results
- Exports results to:
  - Per-run CSV in app/logs/profiles_{session_id}.csv
  - A persistent workbook at the repository root: profiles.xlsx (appends across runs)

Tech Stack
- Python 3.13+ (dependency management via uv)
- OpenAI (default model: gpt-4o-mini) for vision + reasoning
- LangGraph for stateful orchestration
- OpenCV for computer vision (template matching)
- ADB (Android Debug Bridge) for device automation
- Optional: pandas + openpyxl for Excel export

Requirements
- Android device with USB debugging enabled and Hinge installed
- Android SDK Platform Tools (adb)
- Python 3.13+
- uv package manager (recommended)
- OpenAI API key (default). Gemini support is present behind analyzer components.

Setup
1) Create environment file at app/.env:
   OPENAI_API_KEY=your-openai-api-key
   # Optional if you switch implementations:
   GEMINI_API_KEY=your-gemini-api-key

2) Install dependencies (from the app/ directory):
   uv sync

3) Verify device:
   adb devices
   # Your device should be listed

Quick Start (PowerShell)
- Minimal, safer run (1 profile, conservative, verbose, no screenshots):
  Set-Location ".\auto-hinger\app"
  uv run python .\main_agent.py --profiles 1 --config conservative --verbose --no-screenshots

- Standard run (default config, saves screenshots to app/images/):
  Set-Location ".\auto-hinger\app"
  uv run python .\main_agent.py

- With persistent Excel export (XLSX):
  Set-Location ".\auto-hinger\app"
  uv run python .\main_agent.py --profiles 3 --export-xlsx

Command Line Options
- --profiles, -p: Maximum number of profiles to process (default: 10)
- --config, -c: default | fast | conservative (default: default)
- --device-ip: ADB server host (default: 127.0.0.1)
- --verbose, -v: Enable verbose logging
- --no-screenshots: Disable saving screenshots to disk
- --manual-confirm, --confirm-steps: Require explicit confirmation before each step; logs all actions to logs/ (default: disabled)
- --like-mode {priority,normal}: Prefer the send button variant (default: priority)
- --export-xlsx: Also write results into a persistent Excel workbook at repo root (profiles.xlsx)
- --ai-routing: Use AI to route actions instead of the default deterministic safeguards (off by default)

Behavior and Defaults
- Default action is like-with-comment. Like-only is a fallback if typing/sending fails or the comment UI doesn’t appear.
- Priority send is preferred by default; normal send is used if priority is unavailable.
- Startup pre-check: The tool will verify that it starts at the top of the profile feed (Like visible). If not, it exits with a clear message. Navigate to the feed top and re-run.
- Manual confirm before send: enabled by default to allow a final check before tapping Send. You will be prompted in the terminal.

Exports and Where to Find Them
- Per-run CSV: app/logs/profiles_{session_id}.csv
- Persistent Excel workbook (appends across runs): profiles.xlsx at repo root (auto-hinger/profiles.xlsx)
  - XLSX is produced only when --export-xlsx is specified and pandas/openpyxl are installed.

Architecture Overview
- Entry point: app/main_agent.py
  - Parses CLI, applies AgentConfig, instantiates and runs LangGraphHingeAgent.
- Agent core: app/langgraph_hinge_agent.py
  - Nodes include: initialize_session, ai_decide_action, capture_screenshot, analyze_profile, scroll_profile, make_like_decision, detect_like_button, execute_like, generate_comment, send_comment_with_typing, send_like_without_comment, navigate_to_next, verify_profile_change, recover_from_stuck, reset_app, finalize_session.
  - Uses OpenAI for vision/text analysis; OpenCV for template matching; ADB for interactions.
- CV helpers & ADB: app/helper_functions.py
  - detect_like_button_cv, detect_send_button_cv, detect_comment_field_cv; tap/swipe/keyboard helpers; app reset/open.
- Analyzer facade: app/analyzer.py -> app/analyzer_openai.py (default)
  - OCR, UI analysis, comment generation, outcome verification; swappable to Gemini analyzer.
- Data store: app/data_store.py
  - Generated comment store and optional stats.
- Assets: app/assets/
  - like_button.png, comment_field.png, send_button.png, send_priority_button.png (update with device-specific crops if matching is unreliable).
- Images: app/images/ (created at runtime).

Troubleshooting
- Device connection:
  adb devices
  adb kill-server; adb start-server

- Dependencies:
  Set-Location ".\auto-hinger\app"
  uv sync --reinstall

- Run with verbose logs:
  uv run python .\main_agent.py --verbose
