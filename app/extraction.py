import base64
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from llm_client import get_default_model, get_llm_client, resolve_model
from prompts import LLM1_VISUAL, LLM2
from ai_trace import (
    _ai_trace_image_lines,
    _ai_trace_log,
    _ai_trace_log_response,
    _ai_trace_prompt_lines,
)
from runtime import _log
from profile_utils import _get_core
from text_utils import normalize_dashes


def _b64_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_llm_batch_payload(
    screenshots: List[str],
    prompt: Optional[str] = None,
    format: str = "openai_messages",
) -> Dict[str, Any]:
    if not prompt:
        prompt = LLM1_VISUAL()

    existing = [p for p in screenshots if isinstance(p, str) and os.path.exists(p)]

    if format == "openai_messages":
        content_parts = [{"type": "text", "text": prompt}]
        for p in existing:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_b64_image(p)}"},
            })
        return {
            "format": "openai_messages",
            "messages": [{
                "role": "user",
                "content": content_parts,
            }],
            "meta": {
                "images_count": len(existing),
                "images_paths": existing,
            },
        }

    raise ValueError(f"Unsupported format: {format}")


def _default_profile_eval() -> Dict[str, Any]:
    return {
        "home_country_iso": "",
        "home_country_confidence": 0.0,
        "job": {
            "normalized_title": "Unknown",
            "band": "T1",
            "confidence": 0.2,
            "band_reason": "Title empty or unresolved; defaulting to T1 per rules",
        },
        "university_elite": 0,
        "matched_university_name": "",
    }


def run_profile_eval_llm(extracted: Dict[str, Any], model: str | None = None) -> Dict[str, Any]:
    core = _get_core(extracted)
    home_town = core.get("Home town", "") or ""
    job_title = core.get("Job title", "") or ""
    university = core.get("University", "") or ""

    prompt = LLM2(home_town, job_title, university)
    requested_model = model or get_default_model()
    resolved_model = resolve_model(requested_model)
    trace_lines = [
        f"AI_CALL call_id=profile_eval_llm model={resolved_model} response_format=json_object"
    ]
    trace_lines.extend(_ai_trace_prompt_lines(prompt))
    _ai_trace_log(trace_lines)

    try:
        t0 = time.perf_counter()
        resp = get_llm_client().chat.completions.create(
            model=resolved_model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        dt_ms = int((time.perf_counter() - t0) * 1000)
        raw = resp.choices[0].message.content or ""
        try:
            parsed = json.loads(raw or "{}")
        except Exception as e:
            _ai_trace_log_response(
                "profile_eval_llm",
                resolved_model,
                raw,
                parsed=None,
                duration_ms=dt_ms,
                error=f"json_parse_error: {e}",
            )
            _log(f"[LLM2] parse failed: {e}")
            return _default_profile_eval()
        if not isinstance(parsed, dict):
            _ai_trace_log_response(
                "profile_eval_llm",
                resolved_model,
                raw,
                parsed=None,
                duration_ms=dt_ms,
                error="parsed_not_dict",
            )
            return _default_profile_eval()
        parsed = normalize_dashes(parsed)
        _ai_trace_log_response(
            "profile_eval_llm",
            resolved_model,
            raw,
            parsed=parsed,
            duration_ms=dt_ms,
        )
        return parsed
    except Exception as e:
        _ai_trace_log_response(
            "profile_eval_llm",
            resolved_model,
            raw="",
            parsed=None,
            duration_ms=None,
            error=f"call_error: {e}",
        )
        return _default_profile_eval()


def run_llm1_visual(
    image_paths: List[str],
    model: str | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    prompt = LLM1_VISUAL()
    payload = build_llm_batch_payload(image_paths, prompt=prompt)
    requested_model = model or get_default_model()
    resolved_model = resolve_model(requested_model)
    trace_lines = [
        f"AI_CALL call_id=llm1_visual model={resolved_model} response_format=json_object"
    ]
    trace_lines.extend(_ai_trace_prompt_lines(prompt))
    trace_lines.extend(_ai_trace_image_lines(image_paths))
    _ai_trace_log(trace_lines)
    try:
        t0 = time.perf_counter()
        resp = get_llm_client().chat.completions.create(
            model=resolved_model,
            response_format={"type": "json_object"},
            messages=payload.get("messages", []),
        )
        dt_ms = int((time.perf_counter() - t0) * 1000)
        raw = resp.choices[0].message.content or ""
        try:
            parsed = json.loads(raw or "{}")
        except Exception as e:
            _ai_trace_log_response(
                "llm1_visual",
                resolved_model,
                raw,
                parsed=None,
                duration_ms=dt_ms,
                error=f"json_parse_error: {e}",
            )
            _log(f"[LLM1] parse failed: {e}")
            return {}, payload.get("meta", {})
        if not isinstance(parsed, dict):
            _ai_trace_log_response(
                "llm1_visual",
                resolved_model,
                raw,
                parsed=None,
                duration_ms=dt_ms,
                error="parsed_not_dict",
            )
            return {}, payload.get("meta", {})
        _ai_trace_log_response(
            "llm1_visual",
            resolved_model,
            raw,
            parsed=parsed,
            duration_ms=dt_ms,
        )
        return parsed, payload.get("meta", {})
    except Exception as e:
        _ai_trace_log_response(
            "llm1_visual",
            resolved_model,
            raw="",
            parsed=None,
            duration_ms=None,
            error=f"call_error: {e}",
        )
        _log(f"[LLM1] visual call failed: {e}")
        return {}, payload.get("meta", {})


def _build_extracted_profile(
    biometrics: Dict[str, Any],
    ui_map: Dict[str, Any],
    llm1_visual: Dict[str, Any],
) -> Dict[str, Any]:
    core_fields = [
        "Name",
        "Gender",
        "Sexuality",
        "Age",
        "Height",
        "Location",
        "Explicit Ethnicity",
        "Children",
        "Family plans",
        "Covid Vaccine",
        "Pets",
        "Zodiac Sign",
        "Job title",
        "University",
        "Religious Beliefs",
        "Home town",
        "Politics",
        "Languages spoken",
        "Dating Intentions",
        "Relationship type",
        "Drinking",
        "Smoking",
        "Marijuana",
        "Drugs",
    ]
    core: Dict[str, Any] = {k: "" for k in core_fields}
    core["Age"] = None
    core["Height"] = None
    for k, v in biometrics.items():
        if k in {"Age", "Height"}:
            core[k] = v
        elif k in core:
            core[k] = v

    # Prompts
    prompts_out: List[Dict[str, Any]] = []
    prompts = ui_map.get("prompts", [])
    for idx in range(1, 4):
        if idx <= len(prompts):
            p = prompts[idx - 1]
            prompts_out.append(
                {
                    "id": f"prompt_{idx}",
                    "prompt": p.get("prompt", ""),
                    "answer": p.get("answer", ""),
                    "source_file": "",
                    "page_half": "",
                }
            )
        else:
            prompts_out.append(
                {"id": f"prompt_{idx}", "prompt": "", "answer": "", "source_file": "", "page_half": ""}
            )

    # Poll
    poll = ui_map.get("poll", {})
    poll_question = poll.get("question", "") or ""
    poll_answers = poll.get("options", [])
    poll_out = {
        "id": "poll_1",
        "question": poll_question,
        "source_file": "",
        "page_half": "",
        "answers": [
            {"id": "poll_1_a", "text": poll_answers[0].get("text", "") if len(poll_answers) > 0 else ""},
            {"id": "poll_1_b", "text": poll_answers[1].get("text", "") if len(poll_answers) > 1 else ""},
            {"id": "poll_1_c", "text": poll_answers[2].get("text", "") if len(poll_answers) > 2 else ""},
        ],
    }

    # LLM1 visual parsing
    photos_resp = llm1_visual.get("photos", []) if isinstance(llm1_visual, dict) else []
    photo_desc_map: Dict[str, str] = {}
    if isinstance(photos_resp, list):
        for item in photos_resp:
            if isinstance(item, dict):
                pid = item.get("id")
                desc = item.get("description", "")
                if pid:
                    photo_desc_map[pid] = desc
    visual_traits = llm1_visual.get("visual_traits", {}) if isinstance(llm1_visual, dict) else {}
    if not isinstance(visual_traits, dict):
        visual_traits = {}

    photo_entries: List[Dict[str, Any]] = []
    for idx in range(1, 7):
        pid = f"photo_{idx}"
        photo_meta = None
        for p in ui_map.get("photos", []):
            if p.get("id") == pid:
                photo_meta = p
                break
        photo_entries.append(
            {
                "id": pid,
                "description": photo_desc_map.get(pid, ""),
                "source_file": (photo_meta or {}).get("crop_path", ""),
                "page_half": "",
            }
        )

    # Ensure all visual trait keys exist.
    visual_keys = [
        "Face Visibility Quality",
        "Photo Authenticity / Editing Level",
        "Apparent Body Fat Level",
        "Profile Distinctiveness",
        "Apparent Build Category",
        "Apparent Skin Tone",
        "Apparent Ethnic Features",
        "Hair Color",
        "Facial Symmetry Level",
        "Indicators of Fitness or Lifestyle",
        "Overall Visual Appeal Vibe",
        "Apparent Age Range Category",
        "Attire and Style Indicators",
        "Body Language and Expression",
        "Visible Enhancements or Features",
        "Apparent Chest Proportions",
        "Apparent Attractiveness Tier",
        "Reasoning for attractiveness tier",
        "Facial Proportion Balance",
        "Grooming Effort Level",
        "Presentation Red Flags",
        "Visible Tattoo Level",
        "Visible Piercing Level",
        "Short-Term / Hookup Orientation Signals",
    ]
    visual_out = {k: visual_traits.get(k, "") for k in visual_keys}

    return {
        "Core Biometrics (Objective)": core,
        "Profile Content (Free Description)": {
            "Profile Prompts and Answers": prompts_out,
            "Poll (optional, most profiles will not have this)": poll_out,
            "Other text on profile not covered by above": "",
            "Description of any non-photo media (e.g., video (identified via timestamp in top right), voice note)": "",
            "Extensive Description of Photo 1": photo_entries[0],
            "Extensive Description of Photo 2": photo_entries[1],
            "Extensive Description of Photo 3": photo_entries[2],
            "Extensive Description of Photo 4": photo_entries[3],
            "Extensive Description of Photo 5": photo_entries[4],
            "Extensive Description of Photo 6": photo_entries[5],
        },
        "Visual Analysis (Inferred From Images)": {
            "Inferred Visual Traits Summary": visual_out
        },
    }


