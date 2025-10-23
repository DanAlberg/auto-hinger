import os
import base64
from typing import List, Dict, Optional, Any


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
