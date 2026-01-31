# AutoHinge

End-to-end Hinge profile scanner + scorer + opener selector for Android via ADB.

What it does
- Connects to an Android device (ADB)
- Scans a single profile with UI XML + photo crops
- Extracts core biometrics from the UI
- LLM1: visual analysis + photo descriptions
- LLM2: enrichment (home country, job band, university elite)
- Scores long/short
- LLM3: generates openers (long/short)
- LLM4: chooses the best opener + target
- Taps the chosen target like
- Opens the comment field, types the chosen opener, and sends a priority like (confirmation required unless --unrestricted)
- On reject, taps the skip/dislike button (confirmation required unless --unrestricted)

Requirements
- Android device with USB debugging enabled and Hinge installed
- adb on PATH
- Python 3.12+
- uv (optional but recommended)

Setup
1) Create `app/.env`:
   ```
   OPENAI_API_KEY=your-key
   GEMINI_API_KEY=your-key
   LLM_PROVIDER=gemini|openai
   GEMINI_MODEL=gemini-2.5-pro
   GEMINI_SMALL_MODEL=gemini-2.5-flash
   OPENAI_MODEL=gpt-5
   OPENAI_SMALL_MODEL=gpt-5-mini
   ```
   Only the key for your provider is required. `LLM_PROVIDER` defaults to gemini if unset.

2) Install dependencies:
   ```
   cd app
   uv sync
   ```

Run
```
cd app
uv run python start.py
```

Options
- `--unrestricted`: skips confirmations before dislike and send priority like

Outputs
- `profiles.db` at repo root (created on first successful insert)
- `app/images/crops/` photo crops
- `app/logs/` run JSON + score table
- Optional AI trace: set `HINGE_AI_TRACE_FILE=app/logs/ai_trace_YYYYMMDD_HHMMSS.log`
- Optional run JSON echo: set `HINGE_SHOW_RUN_JSON=1`
