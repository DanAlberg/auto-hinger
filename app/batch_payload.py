import os
import base64
from typing import List, Dict, Optional, Any


def _b64_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_profile_prompt() -> str:
    """
    Standardized prompt to extract full profile information from a set of screenshots.
    Transport-agnostic; can be reused across providers.
    """
    return """
    Analyze all provided dating profile screenshots together and extract a thorough, structured summary.

    Return ONLY a strict JSON object with exactly these fields (keys must exist exactly as named; use null or empty arrays/objects when not visible on-screen). Do NOT infer or assume anything that is not explicitly visible on the screenshots.

    {
      "sexuality": "string|null",
      "name": "string|null",                  // REQUIRED KEY (value null if not visible)
      "age": "number|null",                   // REQUIRED KEY (value null if not visible)
      "height": "string|null",                // REQUIRED KEY (value null if not visible)
      "location": "string|null",              // REQUIRED KEY (value null if not visible)
      "ethnicity": "string|null",
      "current_children": "string|null",
      "family_plans": "string|null",          // one of: "Don't want children", "Want children", "Open to children", "Not sure yet", "Prefer not to say"; or null if hidden
      "covid_vaccine": "string|null",
      "pets": {                               // booleans where possible; else null
        "dog": "boolean|null",
        "cat": "boolean|null",
        "bird": "boolean|null",
        "fish": "boolean|null",
        "reptile": "boolean|null"
      },
      "zodiac_sign": "string|null",
      "work": "string|null",                  // company name only (if shown); NOT job title
      "job_title": "string|null",
      "university": "string|null",
      "religious_beliefs": "string|null",
      "hometown": "string|null",
      "politics": "string|null",              // one of: "Liberal", "Moderate", "Conservative", "Not political", "Other", "Prefer not to say"; or null if hidden
      "languages_spoken": ["string", ...],
      "dating_intentions": "string|null",
      "relationship_type": "string|null",

      // Lifestyle fields MUST be exactly one of: "Yes", "Sometimes", "No"; or null if the info is hidden.
      "drinking": "Yes|Sometimes|No|null",
      "smoking": "Yes|Sometimes|No|null",
      "marijuana": "Yes|Sometimes|No|null",
      "drugs": "Yes|Sometimes|No|null",

      "bio": "string|null",
      "prompts_and_answers": [
        {"prompt": "string", "answer": "string"}
      ],
      "interests": ["string", ...],
      "summary": "short free-text overview"
    }

    Guidance:
    - REQUIRED KEYS: name, age, height, location MUST be present at top-level; set to null if not visible.
    - All other fields are optional and may be null if not visible.
    - Use only explicit on-screen text/chips/labels; if uncertain, set the field to null.
    - For lifestyle fields, do not assume: if not shown, set null; otherwise map to exactly "Yes", "Sometimes", or "No".
    - For family_plans and politics, restrict to the enumerations above; if not shown, set null.
    - "work" is the company name only if explicitly shown; if absent, set null (do not reuse job_title).
    - Output ONLY a valid JSON object with no commentary, preamble, markdown, or code fences.
    """


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
