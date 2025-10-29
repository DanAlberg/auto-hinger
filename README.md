# auto-hinger — Hinge Automation

Automates interactions in the Hinge Android app using ADB, computer-vision template matching, and an optional cloud vision/text analysis provider. Navigates profiles, extracts content, makes decisions, and exports results.

Warning
- This can send automated likes/comments. Review platform terms and use at your own risk.
- Screenshots may be analyzed by an external provider if enabled. Do not run with private data you’re unwilling to share.
- Use a test account and enable step-by-step confirmations when not in scrape-only.

Requirements
- Android device with USB debugging enabled and Hinge installed
- Android SDK Platform Tools (adb) on PATH
- Python 3.12+
- uv package manager
- Provider API key (env var used by the app; see Setup)

Setup
1) Environment (from repo root)
   Create app/.env:
   ```
   OPENAI_API_KEY=your-provider-api-key
   ```

2) Install dependencies (from the app/ directory):
   ```
   cd app
   uv sync
   ```

3) Verify device:
   ```
   adb devices
   # your device should be listed as "device"
   ```

Quick start
```
cd app
uv sync
uv run python main_agent.py
```

Launch options
- Standard run (default config, saves screenshots to app/images/):
  ```
  cd app
  uv run python main_agent.py
  ```

- Dry-run (exercise full flow, skip LIKE/SEND taps):
  ```
  cd app
  uv run python main_agent.py --profiles 1 --dry-run
  ```

- Scrape-only (no likes/dislikes; collect screenshots + extract only):
  ```
  cd app
  uv run python main_agent.py --profiles 1 --scrape-only
  ```

- Safer + verbose with confirmations:
  ```
  cd app
  uv run python main_agent.py --profiles 1 --verbose --confirm-steps
  ```

CLI options
- --profiles, -p               Maximum number of profiles to process (default: 10)
- --config, -c                 default | fast (default: default)
- --device-ip                  ADB server host (default: 127.0.0.1)
- --verbose, -v                Enable verbose logging
- --manual-confirm, --confirm-steps  Require confirmation before each step; logs actions to logs/ (default: disabled)
- --like-mode {priority,normal} Prefer the send button variant (default: priority)
- --no-excel, --no-xlsx        Disable Excel workbook logging
- --ai-routing                 Enable an alternative routing mode (off by default)
- --dry-run                    Run full logic but skip LIKE/SEND taps
- --scrape-only                Scrape-only mode; no like/dislike actions
- --skip-precheck, --no-precheck  Bypass startup like-button pre-check

Behavior and defaults
- Default action is like-with-comment. Falls back to like-only if typing/sending fails or the comment UI doesn’t appear.
- Priority send preferred by default; normal send used if priority is unavailable.
- Startup pre-check ensures the app begins at the top of the Hinge feed (Like visible); otherwise exits with guidance. For testing, you can bypass this with --skip-precheck.
- Manual confirm is off by default; enable with --confirm-steps for safety when testing.

Outputs
- Excel workbook: profiles.xlsx at repository root
  - Updated incrementally; consecutive-duplicate guard: name+age+height
- Screenshots: app/images/
- Logs: app/logs/

Architecture
- Entry point: app/main_agent.py
  - Parses CLI flags, loads AgentConfig, and launches the orchestrator.
- Orchestration (state machine): app/langgraph_hinge_agent.py
  - Manages session state and routes actions through nodes:
    - initialize_session → capture_screenshot → analyze_profile → scroll_profile
    - make_like_decision → detect_like_button → execute_like
    - send_comment_with_typing / send_like_without_comment
    - navigate_to_next → verify_profile_change → finalize_session
    - recover_from_stuck and reset_app for resilience
  - Supports deterministic flow as well as an alternative routing mode.
- Computer vision & device I/O: app/helper_functions.py
  - Template matching for UI elements (like/send/comment field)
  - ADB wrappers for taps, swipes, keyboard input, and app lifecycle
- Analysis provider facade: app/analyzer.py (with a default implementation in app/analyzer_openai.py)
  - Handles OCR/extraction, UI interpretation, and decision hints
  - Provider is selected/configured via environment variables
- Export & data: app/profile_export.py, app/data_store.py
  - Writes to profiles.xlsx using openpyxl (incremental updates + de-dup guard)
- Configuration: app/agent_config.py
  - Centralized runtime flags and mode toggles
- Assets: app/assets/
  - Template images for CV matching (like_button.png, comment_field.png, send_button.png, send_priority_button.png)
  - Swap with device-specific crops if matching is unreliable

Troubleshooting
- Device connection:
  ```
  adb devices
  adb kill-server
  adb start-server
  ```

- Dependencies:
  ```
  cd app
  uv sync --reinstall
  ```

- Verbose run:
  ```
  cd app
  uv run python main_agent.py --verbose
