import os
import base64
import sqlite3
import json
from typing import List, Dict, Optional, Any

from sqlite_store import get_db_path, update_profile_opener_fields
import prompt_engine
import analyzer_openai


def _b64_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_profile_prompt() -> str:
    """
    Build the structured extraction prompt for the new main LLM call.
    """
    return (
        "You are analyzing screenshots of a dating profile. Each image may contain text, icons, or structured fields. "
        "Your task is to extract only the information that is explicitly visible on-screen. "
        "If a field is not directly stated, leave it empty (do not infer or guess).\n\n"
        "Return a single valid JSON object with the following fields:\n\n"
        "{\n"
        '  "Name": "",\n'
        '  "Gender": "",\n'
        '  "Sexuality": "",\n'
        '  "Age": 0,\n'
        '  "Height": 0,\n'
        '  "Location": "",\n'
        '  "Ethnicity": "",\n'
        '  "Children": "",\n'
        '  "Family plans": "",\n'
        '  "Covid Vaccine": "",\n'
        '  "Pets": "",\n'
        '  "Zodiac Sign": "",\n'
        '  "Job title": "",\n'
        '  "University": "",\n'
        '  "Religious Beliefs": "",\n'
        '  "Home town": "",\n'
        '  "Politics": "",\n'
        '  "Languages spoken": "",\n'
        '  "Dating Intentions": "",\n'
        '  "Relationship type": "",\n'
        '  "Drinking": "",\n'
        '  "Smoking": "",\n'
        '  "Marijuana": "",\n'
        '  "Drugs": "",\n'
        '  "Profile Prompts and Answers": [\n'
        '    {"prompt": "", "answer": ""},\n'
        '    {"prompt": "", "answer": ""},\n'
        '    {"prompt": "", "answer": ""}\n'
        '  ],\n'
        '  "Other text on profile not covered by above": ""\n'
        "}\n\n"
        "Rules:\n"
        "- If a field is not visible, leave it empty or null.\n"
        "- For tri-state fields (Drinking, Smoking, Marijuana, Drugs), only use 'Yes', 'Sometimes', or 'No'.\n"
        "- For categorical fields, use only the following valid options:\n"
        "  • Children: 'Don't have children', 'Have children'\n"
        "  • Family plans: 'Don't want children', 'Want children', 'Not sure yet'\n"
        "  • Covid Vaccine: 'Vaccinated', 'Partially vaccinated', 'Not yet vaccinated'\n"
        "  • Dating Intentions: 'Life partner', 'Long-term relationship', 'Long-term relationship, open to short', 'Short-term relationship, open to long', 'Short term relationship', 'Figuring out my dating goals'\n"
        "  • Relationship type: 'Monogamy', 'Non-Monogamy', 'Figuring out my relationship type'\n"
        "- For tri-state lifestyle fields (Drinking, Smoking, Marijuana, Drugs), only use 'Yes', 'Sometimes', or 'No'.\n"
        "- For 'Profile Prompts and Answers', extract up to three visible prompt/answer pairs.\n"
        "- For 'Other text', include any visible text that doesn’t fit the above categories (e.g., subtext, taglines, or captions).\n\n"
        "Return only the JSON object. Do not include commentary, markdown, or code fences."
        "One of the images will be a stitched image containing biometrics. This is part of the profile and should be included in the results"
    )


def build_llm_batch_payload(
    screenshots: List[str],
    prompt: Optional[str] = None,
    format: str = "openai_messages"
) -> Dict[str, Any]:
    """
    Build a transport-agnostic payload for submitting multiple screenshots to an LLM.

    Supported:
    - format="openai_messages": returns:
      {
        "format": "openai_messages",
        "messages": [{
          "role": "user",
          "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            ...
          ]
        }],
        "meta": { "images_count": N, "images_paths": [...] }
      }

    Note: This function does NOT perform any network calls. It only builds payloads.
    """
    if not prompt:
        prompt = build_profile_prompt()

    # Filter to existing files only, preserve order
    existing = [p for p in screenshots if isinstance(p, str) and os.path.exists(p)]

    if format == "openai_messages":
        content_parts = [{"type": "text", "text": prompt}]
        for p in existing:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_b64_image(p)}"}
            })
        return {
            "format": "openai_messages",
            "messages": [{
                "role": "user",
                "content": content_parts
            }],
            "meta": {
                "images_count": len(existing),
                "images_paths": existing
            }
        }

    # Fallback generic structure for future adapters
    return {
        "format": "unknown",
        "prompt": prompt,
        "images_b64": [_b64_image(p) for p in existing],
        "meta": {
            "images_count": len(existing),
            "images_paths": existing
        }
    }


# ---------------- Opening-style helpers (DB → prompt payload) ----------------

_PROFILE_FIELDS = [
    "id",
    "Name",
    "Gender",
    "Sexuality",
    "Age",
    "Height_cm",
    "Location",
    "Ethnicity",
    "Children",
    "Family_plans",
    "Covid_vaccine",
    "Pets",
    "Zodiac_Sign",
    "Job_title",
    "University",
    "Religious_Beliefs",
    "Home_town",
    "Politics",
    "Languages_spoken",
    "Dating_Intentions",
    "Relationship_type",
    "Drinking",
    "Smoking",
    "Marijuana",
    "Drugs",
    "prompt_1",
    "answer_1",
    "prompt_2",
    "answer_2",
    "prompt_3",
    "answer_3",
    "Other_text",
    "Media_description",
    "Photo1_desc",
    "Photo2_desc",
    "Photo3_desc",
    "Photo4_desc",
    "Photo5_desc",
    "Photo6_desc",
    # Opener style fields (for building opening_messages prompt)
    "primary_style",
    "flirty",
    "complimentary",
    "playful_witty",
    "observational",
    "shared_interest",
    "genuinely_warm",
    "relationship_forward",
    "overall_confidence",
    "rationale",
    # Opening messages storage (JSON blob)
    "opening_messages_json",
]


def _connect(_db_path: Optional[str] = None) -> sqlite3.Connection:
    dbp = _db_path or get_db_path()
    con = sqlite3.connect(dbp)
    con.row_factory = sqlite3.Row
    return con


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def fetch_profile_by_id(profile_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    con = _connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            f"SELECT {', '.join(_PROFILE_FIELDS)} FROM profiles WHERE id = ? LIMIT 1;",
            (int(profile_id),),
        )
        row = cur.fetchone()
        return _row_to_dict(row) if row else None
    finally:
        con.close()


def fetch_latest_profiles(limit: int = 1, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    limit = max(1, int(limit))
    con = _connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            f"SELECT {', '.join(_PROFILE_FIELDS)} FROM profiles ORDER BY datetime(timestamp) DESC, id DESC LIMIT ?;",
            (limit,),
        )
        rows = cur.fetchall() or []
        return [_row_to_dict(r) for r in rows]
    finally:
        con.close()


def build_profile_for_opening_style(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare a profile dict keyed as expected by prompt_engine.render_opening_style_user_message.
    Renderer will omit empty values when composing the prompt.
    """
    if not isinstance(row, dict):
        row = {}
    out = {k: row.get(k) for k in _PROFILE_FIELDS if k in row}

    # Coerce numeric-friendly fields
    for k in ("Age", "Height_cm"):
        try:
            if out.get(k) is not None and str(out[k]).strip() != "":
                out[k] = int(float(out[k]))
        except Exception:
            pass

    # Strip strings
    for k, v in list(out.items()):
        if isinstance(v, str):
            out[k] = v.strip()

    return out


def get_profile_payload_by_id(profile_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    row = fetch_profile_by_id(profile_id, db_path=db_path)
    if not row:
        return None
    return build_profile_for_opening_style(row)


def get_latest_profile_payload(limit: int = 1, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    rows = fetch_latest_profiles(limit=limit, db_path=db_path)
    return [build_profile_for_opening_style(r) for r in rows]


def run_opening_style(profile_id: Optional[int] = None, db_path: Optional[str] = None, model: str = "gpt-5-mini") -> Dict[str, Any]:
    """
    Execute the opening-style LLM request for a given profile id (or latest if None),
    print the JSON to console, and persist fields onto the profiles row.
    """
    # Select source row (keep id for DB update)
    row: Optional[Dict[str, Any]]
    if profile_id is not None:
        row = fetch_profile_by_id(int(profile_id), db_path=db_path)
    else:
        latest = fetch_latest_profiles(limit=1, db_path=db_path)
        row = latest[0] if latest else None

    if not row:
        print("[opening_style] No profile row found.")
        return {}

    pid = int(row["id"])
    profile_payload = build_profile_for_opening_style(row)

    # Build prompts
    system_prompt, user_prompt = prompt_engine.build_opening_style_prompts(profile_payload)

    # Call LLM (JSON-only)
    result = analyzer_openai.chat_json_system_user(system_prompt, user_prompt, model=model)

    # Persist flattened weights and other fields to DB
    update_profile_opener_fields(pid, result, db_path=db_path)

    return result


# ---------------- Opening-messages (uses scraped profile + opening_style result) ----------------

def _build_profile_json_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Compact JSON reflecting scraped content, for embedding in the opening-messages prompt."""
    def _v(k): 
        v = row.get(k)
        if isinstance(v, str):
            return v.strip()
        return v
    # Height as N cm string
    height_cm = row.get("Height_cm")
    height_str = None
    try:
        if height_cm is not None and str(height_cm).strip() != "":
            hv = int(float(height_cm))
            if hv > 0:
                height_str = f"{hv} cm"
    except Exception:
        height_str = None

    # Prompts array
    prompts = []
    for i in (1, 2, 3):
        p = (_v(f"prompt_{i}") or "")
        a = (_v(f"answer_{i}") or "")
        if p or a:
            prompts.append({"prompt": p, "answer": a})

    # Photos
    photos = []
    for i in (1, 2, 3, 4, 5, 6):
        desc = _v(f"Photo{i}_desc") or ""
        if desc:
            photos.append({"index": i, "description": desc})

    prof = {
        "Name": _v("Name"),
        "Gender": _v("Gender"),
        "Sexuality": _v("Sexuality"),
        "Age": _v("Age"),
        "Height": height_str,
        "Location": _v("Location"),
        "Ethnicity": _v("Ethnicity"),
        "Children": _v("Children"),
        "Family plans": _v("Family_plans"),
        "Covid Vaccine": _v("Covid_vaccine"),
        "Pets": _v("Pets"),
        "Zodiac Sign": _v("Zodiac_Sign"),
        "Job title": _v("Job_title"),
        "University": _v("University"),
        "Religious Beliefs": _v("Religious_Beliefs"),
        "Home town": _v("Home_town"),
        "Politics": _v("Politics"),
        "Languages spoken": _v("Languages_spoken"),
        "Dating Intentions": _v("Dating_Intentions"),
        "Relationship type": _v("Relationship_type"),
        "Drinking": _v("Drinking"),
        "Smoking": _v("Smoking"),
        "Marijuana": _v("Marijuana"),
        "Drugs": _v("Drugs"),
        "Profile Prompts and Answers": prompts,
        "Other text on profile not covered by above": _v("Other_text"),
        "Description of any non-photo media (For example poll, voice note)": _v("Media_description"),
        "Photos": photos,
    }
    return prof


def _build_opening_style_json_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Construct opening_style JSON from saved columns for use in the opening-messages prompt."""
    import json
    def _float(x):
        try:
            return float(x)
        except Exception:
            return 0.0

    style = {
        "primary_style": (row.get("primary_style") or "").strip(),
        "style_weights": {
            "flirty": _float(row.get("flirty")),
            "complimentary": _float(row.get("complimentary")),
            "playful_witty": _float(row.get("playful_witty")),
            "observational": _float(row.get("observational")),
            "shared_interest": _float(row.get("shared_interest")),
            "genuinely_warm": _float(row.get("genuinely_warm")),
            "relationship_forward": _float(row.get("relationship_forward")),
        },
        "overall_confidence": _float(row.get("overall_confidence")),
        "rationale": row.get("rationale") or ""
    }
    return style


def run_opening_messages(profile_id: Optional[int] = None, db_path: Optional[str] = None, model: str = "gpt-4o") -> Dict[str, Any]:
    """
    Execute the opening-messages LLM request for a given profile id (or latest if None).
    Uses scraped profile JSON + opening-style JSON; persists full JSON to opening_messages_json.
    """
    # Select source row
    row: Optional[Dict[str, Any]]
    if profile_id is not None:
        row = fetch_profile_by_id(int(profile_id), db_path=db_path)
    else:
        latest = fetch_latest_profiles(limit=1, db_path=db_path)
        row = latest[0] if latest else None

    if not row:
        print("[opening_messages] No profile row found.")
        return {}

    pid = int(row["id"])
    profile_json = _build_profile_json_from_row(row)
    opening_style_json = _build_opening_style_json_from_row(row)

    # Build prompts
    system_prompt, user_prompt = prompt_engine.build_opening_messages_prompts(profile_json, opening_style_json)

    # Call LLM (JSON-only) with gpt-5 as requested
    result = analyzer_openai.chat_json_system_user(system_prompt, user_prompt, model=model)

    # Persist JSON blob
    from sqlite_store import update_profile_opening_messages_json
    update_profile_opening_messages_json(pid, result, db_path=db_path)

    return result


def run_opening_pick(profile_id: Optional[int] = None, db_path: Optional[str] = None, model: str = "gpt-5") -> Dict[str, Any]:
    """
    Execute the opening-pick LLM request for a given profile id (or latest if None).
    Requires opening_messages_json to be present on the row.
    Persists the full selection JSON and chosen_text into dedicated columns.
    """
    # Select source row
    row: Optional[Dict[str, Any]]
    if profile_id is not None:
        row = fetch_profile_by_id(int(profile_id), db_path=db_path)
    else:
        latest = fetch_latest_profiles(limit=1, db_path=db_path)
        row = latest[0] if latest else None

    if not row:
        print("[opening_pick] No profile row found.")
        return {}

    pid = int(row["id"])

    # Parse generated openers from previous step
    import json as _json
    generated_openers_json = {}
    try:
        raw = row.get("opening_messages_json") or "{}"
        generated_openers_json = _json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception as _e:
        print(f"[opening_pick] invalid opening_messages_json: {_e}")
        generated_openers_json = {}

    if not isinstance(generated_openers_json, dict) or not isinstance(generated_openers_json.get("openers"), list) or len(generated_openers_json["openers"]) == 0:
        print("[opening_pick] Missing or empty generated openers; skipping pick.")
        return {}

    # Build inputs
    profile_json = _build_profile_json_from_row(row)
    opening_style_json = _build_opening_style_json_from_row(row)

    # Prompts
    system_prompt, user_prompt = prompt_engine.build_opening_pick_prompts(profile_json, opening_style_json, generated_openers_json)

    # Call LLM (JSON-only) with gpt-5
    result = analyzer_openai.chat_json_system_user(system_prompt, user_prompt, model=model)

    # Persist selection (full JSON + chosen_text)
    from sqlite_store import update_profile_opening_pick
    update_profile_opening_pick(pid, result, db_path=db_path)

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run opening-style analysis for a profile (latest by default).")
    parser.add_argument("--id", type=int, default=None, help="Profile id to analyse. If omitted, uses latest.")
    parser.add_argument("--db", type=str, default=None, help="Explicit path to profiles.db (optional).")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="Model name.")
    args = parser.parse_args()
    out = run_opening_style(profile_id=args.id, db_path=args.db, model=args.model)
    try:
        print("[opening_style result]")
        print(json.dumps(out, indent=2))
    except Exception:
        pass
