# app/profile_eval.py
# Evaluates extracted profile fields (Home town, Job title, University) via GPT-5
# Returns enforced-JSON with modifiers per provided rules.

import os
import json
import time
from typing import Any, Dict
from openai import OpenAI
import config  # ensure .env is loaded at import time

from prompt_engine import build_profile_eval_prompt

# Initialize OpenAI client (reads OPENAI_API_KEY from environment)
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Minimal AI trace helpers (inputs only), reusing same env knobs as analyzer_openai.py
from datetime import datetime

def _ai_trace_file() -> str:
    return os.getenv("HINGE_AI_TRACE_FILE", "")

def _ai_trace_console() -> bool:
    return os.getenv("HINGE_AI_TRACE_CONSOLE", "") == "1"

def _ai_trace_enabled() -> bool:
    return bool(_ai_trace_file())

def _ai_trace_log(lines):
    if not _ai_trace_enabled():
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    out_lines = [f"[{ts}] {line}" for line in lines]
    try:
        with open(_ai_trace_file(), "a", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
    except Exception:
        pass
    if _ai_trace_console():
        for l in out_lines:
            try:
                print(l)
            except Exception:
                pass


def _default_profile_eval() -> Dict[str, Any]:
    # Neutral default per rules (unknowns, zeros, conservative)
    return {
        "home_country_iso": "",
        "home_country_confidence": 0.0,
        "home_country_modifier": 0,
        "job": {
            "normalized_title": "Unknown",
            "est_salary_gbp": 0,
            "band": "B2",
            "confidence": 0.2,
            "band_reason": "Title empty or unresolved; defaulting to B2 per rules"
        },
        "job_modifier": 0,
        "university_elite": 0,
        "matched_university_name": "",
        "university_modifier": 0
    }


def evaluate_profile_fields(extracted: Dict[str, Any], model: str | None = None) -> Dict[str, Any]:
    """
    Evaluate fields from structured extraction:
      - Home town (string)
      - Job title (string)
      - University (string)
    Returns exactly one dict per the specified JSON schema with final numeric modifiers computed.
    """
    if not isinstance(extracted, dict):
        extracted = {}

    home_town = extracted.get("Home town", "") or ""
    job_title = extracted.get("Job title", "") or ""
    university = extracted.get("University", "") or ""

    prompt = build_profile_eval_prompt(home_town, job_title, university)

    # AI trace input (prompt only)
    _ai_trace_log([
        f"AI_CALL call_id=evaluate_profile_fields model={model or 'gpt-5'} response_format=json_object temperature=0.0",
        "PROMPT=<<<BEGIN",
        *prompt.splitlines(),
        "<<<END",
    ])

    try:
        t0 = time.perf_counter()
        resp = _client.chat.completions.create(
            model=(model or "gpt-5"),
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}]
        )
        dt_ms = int((time.perf_counter() - t0) * 1000)
        try:
            print(f"[AI] evaluate_profile_fields model={model or 'gpt-5'} duration={dt_ms}ms")
        except Exception:
            pass
        _ai_trace_log([f"AI_TIME call_id=evaluate_profile_fields model={model or 'gpt-5'} duration_ms={dt_ms}"])

        raw = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        # Print preview to console (truncated)
        try:
            print("[AI JSON profile_eval]")
            print(json.dumps(parsed, indent=2)[:2000])
        except Exception:
            pass
        return parsed if isinstance(parsed, dict) else _default_profile_eval()
    except Exception as e:
        try:
            print(f"⚠️ evaluate_profile_fields failed: {e}")
        except Exception:
            pass
        return _default_profile_eval()
