# app/profile_eval.py
# Evaluates extracted profile fields (Home town, Job title, University) via GPT-5
# Returns enforced-JSON with modifiers per provided rules.

import os
import json
import time
from typing import Any, Dict, List, Tuple
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


# ===== Scoring logic =====

NUKE_SCORE = -99
LIKE_THRESHOLD = 3  # 3+ = LIKE, 2- = DISLIKE

SHORT_TERM_SET = {
    "Short term relationship",
    "Short-term relationship, open to long",
}

def _val(d: Dict[str, Any], key: str, default: Any = "") -> Any:
    v = d.get(key, default)
    return v if v is not None else default

def _add(contribs: List[Tuple[str, str, int]], field: str, value: str, delta: int):
    if delta != 0:
        contribs.append((field, value, int(delta)))

def _check_nukes(extracted: Dict[str, Any]) -> List[str]:
    nukes: List[str] = []

    children = _val(extracted, "Children", "")
    if children == "Have children":
        nukes.append('Children="Have children"')

    religion = _val(extracted, "Religious Beliefs", "")
    if religion == "Muslim":
        nukes.append('Religious Beliefs="Muslim"')

    smoking = _val(extracted, "Smoking", "")
    if smoking in ("Yes", "Sometimes"):
        nukes.append(f'Smoking="{smoking}"')

    marijuana = _val(extracted, "Marijuana", "")
    if marijuana == "Yes":
        nukes.append('Marijuana="Yes"')

    drugs = _val(extracted, "Drugs", "")
    if drugs == "Yes":
        nukes.append('Drugs="Yes"')

    # Age > 40 unless short-term intentions
    age = _val(extracted, "Age", 0) or 0
    di = _val(extracted, "Dating Intentions", "")
    if isinstance(age, int) and age > 40 and di not in SHORT_TERM_SET:
        nukes.append("Age>40 without short-term intentions")

    return nukes

def _score_profile(extracted: Dict[str, Any], enrichment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns:
      {
        "decision": "LIKE"|"DISLIKE",
        "score": int,
        "nukes_triggered": [str],
        "top_contributors": [{"field":..., "value":..., "delta":...}, ...]
      }
    """
    nukes = _check_nukes(extracted)
    if nukes:
        result = {
            "decision": "DISLIKE",
            "score": NUKE_SCORE,
            "nukes_triggered": nukes,
            "top_contributors": []
        }
        print("[SCORER] ", result)
        return result

    score = 0
    contribs: List[Tuple[str, str, int]] = []

    # Sexuality
    sexuality = _val(extracted, "Sexuality", "")
    if sexuality == "Bisexual":
        score += 1
        _add(contribs, "Sexuality", sexuality, +1)

    # Zodiac
    zodiac = _val(extracted, "Zodiac Sign", "")
    if str(zodiac).strip() != "":
        score -= 1
        _add(contribs, "Zodiac Sign", str(zodiac), -1)

    # Job (from enrichment)
    job_mod = int(enrichment.get("job_modifier", 0))
    job_band = enrichment.get("job", {}).get("band", "")
    score += job_mod
    _add(contribs, "Job", f'band={job_band}', job_mod)

    # University (from enrichment)
    uni_mod = int(enrichment.get("university_modifier", 0))
    if uni_mod:
        score += uni_mod
        _add(contribs, "University", "elite=1", uni_mod)

    # Religion (+1 atheist/agnostic/none/empty, +1 Jewish)
    religion = _val(extracted, "Religious Beliefs", "")
    if religion in ("Atheist", "Agnostic", "None", ""):
        score += 1
        _add(contribs, "Religious Beliefs", religion or "(empty)", +1)
    if religion == "Jewish":
        score += 1
        _add(contribs, "Religious Beliefs", "Jewish", +1)

    # Home country (from enrichment; GB already neutral in enrichment rules)
    home_mod = int(enrichment.get("home_country_modifier", 0))
    if home_mod:
        iso = enrichment.get("home_country_iso", "")
        score += home_mod
        _add(contribs, "Home country", iso or "(unresolved)", home_mod)

    # Politics
    politics = _val(extracted, "Politics", "")
    if politics == "Liberal":
        score -= 1
        _add(contribs, "Politics", "Liberal", -1)
    elif politics == "Conservative":
        score += 1
        _add(contribs, "Politics", "Conservative", +1)
    elif politics == "Not political":
        score += 1
        _add(contribs, "Politics", "Not political", +1)

    # Age buckets
    age = _val(extracted, "Age", 0) or 0
    if isinstance(age, int) and age > 0:
        if 18 <= age <= 25:
            score += 4
            _add(contribs, "Age", "18-25", +4)
        elif 26 <= age <= 27:
            score += 3
            _add(contribs, "Age", "26-27", +3)
        elif 28 <= age <= 30:
            _add(contribs, "Age", "28-30", 0)
        elif 31 <= age <= 35:
            score -= 2
            _add(contribs, "Age", "31-35", -2)
        elif 36 <= age <= 40:
            score -= 3
            _add(contribs, "Age", "36-40", -3)

    # Dating Intentions
    di = _val(extracted, "Dating Intentions", "")
    if di == "Short term relationship":
        score += 2
        _add(contribs, "Dating Intentions", di, +2)
    elif di == "Short-term relationship, open to long":
        score += 2
        _add(contribs, "Dating Intentions", di, +2)
    elif di == "Figuring out my dating goals":
        score += 1
        _add(contribs, "Dating Intentions", di, +1)
    elif di == "Life partner":
        score -= 1
        _add(contribs, "Dating Intentions", di, -1)

    # Drinking
    drinking = _val(extracted, "Drinking", "")
    if drinking == "Sometimes":
        score += 1
        _add(contribs, "Drinking", "Sometimes", +1)

    # Marijuana / Drugs "Sometimes"
    marijuana = _val(extracted, "Marijuana", "")
    if marijuana == "Sometimes":
        score -= 1
        _add(contribs, "Marijuana", "Sometimes", -1)

    drugs = _val(extracted, "Drugs", "")
    if drugs == "Sometimes":
        score -= 1
        _add(contribs, "Drugs", "Sometimes", -1)

    # Height (>175 => +1)
    height = _val(extracted, "Height", 0) or 0
    if isinstance(height, int) and height > 175:
        score += 1
        _add(contribs, "Height", f"{height}", +1)

    # Ethnicity
    ethnicity = _val(extracted, "Ethnicity", "")
    if ethnicity == "Black/African Descent":
        score -= 3
        _add(contribs, "Ethnicity", ethnicity, -3)
    elif ethnicity == "Southeast Asian":
        score -= 2
        _add(contribs, "Ethnicity", ethnicity, -2)

    # Decision
    decision = "LIKE" if score >= LIKE_THRESHOLD else "DISLIKE"

    # Top contributors by absolute magnitude (up to 5)
    contribs_sorted = sorted(contribs, key=lambda c: abs(c[2]), reverse=True)[:5]
    top_contributors = [{"field": f, "value": v, "delta": d} for (f, v, d) in contribs_sorted]

    result = {
        "decision": decision,
        "score": int(score),
        "nukes_triggered": [],
        "top_contributors": top_contributors
    }
    print("[SCORER] ", result)
    return result


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
        f"AI_CALL call_id=evaluate_profile_fields model={model or 'gpt-5'} response_format=json_object",
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
        if not isinstance(parsed, dict):
            parsed = _default_profile_eval()
        # Print preview to console (truncated)
        try:
            print("[AI JSON profile_eval]")
            print(json.dumps(parsed, indent=2)[:2000])
        except Exception:
            pass
        # Run scoring using extracted (structured JSON) + enrichment (parsed)
        try:
            scoring = _score_profile(extracted, parsed)
            try:
                print(f"[SCORER_SUMMARY] decision={scoring.get('decision')} score={int(scoring.get('score', 0))}")
            except Exception:
                pass
        except Exception as _se:
            try:
                print(f"⚠️ scoring failed: {_se}")
            except Exception:
                pass
        return parsed
    except Exception as e:
        try:
            print(f"⚠️ evaluate_profile_fields failed: {e}")
        except Exception:
            pass
        return _default_profile_eval()
