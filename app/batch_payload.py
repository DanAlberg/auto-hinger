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

    Return JSON with exactly these fields (use null or empty arrays/objects when unknown or not present):

    {
      "sexuality": "string|null",
      "name": "string|null",
      "age": "number|null",
      "height": "string|null",
      "location": "string|null",
      "ethnicity": "string|null",
      "current_children": "string|null",
      "family_plans": "string|null",           // future children; e.g., "wants kids", "doesn't want"
      "covid_vaccine": "string|null",
      "pets": {                                // booleans where possible; else null
        "dog": "boolean|null",
        "cat": "boolean|null",
        "bird": "boolean|null",
        "fish": "boolean|null",
        "reptile": "boolean|null"
      },
      "zodiac_sign": "string|null",
      "work": "string|null",                   // company or general work descriptor
      "job_title": "string|null",
      "university": "string|null",
      "education_level": "string|null",
      "religious_beliefs": "string|null",
      "hometown": "string|null",
      "politics": "string|null",
      "languages_spoken": ["string", ...],
      "dating_intentions": "string|null",
      "relationship_type": "string|null",
      "drinking": "string|null",
      "smoking": "string|null",
      "marijuana": "string|null",
      "drugs": "string|null",

      "bio": "string|null",
      "prompts_and_answers": [
        {"prompt": "string", "answer": "string"}
      ],
      "interests": ["string", ...],
      "summary": "short free-text overview"
    }

    Guidance:
    - Use all images collectively to infer details (name, age, bio, prompts/answers, chips/labels).
    - Prefer explicit text on screenshots over inference; if not explicit, leave null.
    - Normalize short textual values where possible (e.g., "Socially" for drinking).
    - Keep JSON valid and concise, no additional commentary outside the JSON.
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
