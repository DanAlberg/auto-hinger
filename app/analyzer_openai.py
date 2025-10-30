# app/analyzer_openai.py
# OpenAI-backed analyzer implementations (facade-compatible signatures)
# so existing call sites remain stable.

import os
import json
import base64
from typing import Optional, Dict, Any, List
from openai import OpenAI
import config  # ensure .env is loaded at import time
import time


# Initialize OpenAI client (reads OPENAI_API_KEY from environment)
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- AI trace helpers (inputs only) ---
from datetime import datetime

def _ai_trace_file() -> str:
    return os.getenv("HINGE_AI_TRACE_FILE", "")

def _ai_trace_console() -> bool:
    return os.getenv("HINGE_AI_TRACE_CONSOLE", "") == "1"

def _ai_trace_enabled() -> bool:
    return bool(_ai_trace_file())

def _ai_trace_log(lines):
    """Write trace lines with timestamp to the configured file and optionally to console."""
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
            print(l)


def _b64_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _chat_text(prompt: str, temperature: float = 0.2, model: str = "gpt-5-mini") -> str:
    # AI trace: log the prompt exactly as sent (no images here)
    _ai_trace_log([
        f"AI_CALL call_id=chat_text model={model} temperature={temperature}",
        "PROMPT=<<<BEGIN",
        *prompt.splitlines(),
        "<<<END",
    ])
    t0 = time.perf_counter()
    resp = _client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    dt_ms = int((time.perf_counter() - t0) * 1000)
    _ai_trace_log([f"AI_TIME call_id=chat_text model={model} duration_ms={dt_ms}"])
    try:
        print(f"[AI] chat_text model={model} duration={dt_ms}ms")
    except Exception:
        pass
    content = resp.choices[0].message.content or ""
    return content.strip()


def _chat_json(prompt: str, image_path: Optional[str] = None, temperature: float = 0.0, model: str = "gpt-5-mini") -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = []
    if image_path:
        # AI trace: log prompt and image path + size (never log base64)
        try:
            _sz = os.path.getsize(image_path)
        except Exception:
            _sz = "?"
        _ai_trace_log([
            f"AI_CALL call_id=chat_json model={model} response_format=json_object",
            "PROMPT=<<<BEGIN",
            *prompt.splitlines(),
            "<<<END",
            f"IMAGE image_path={image_path} image_size={_sz} bytes"
        ])
        b64 = _b64_image(image_path)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
        })
    else:
        _ai_trace_log([
            f"AI_CALL call_id=chat_json model={model} response_format=json_object",
            "PROMPT=<<<BEGIN",
            *prompt.splitlines(),
            "<<<END",
        ])
        messages.append({"role": "user", "content": prompt})

    t0 = time.perf_counter()
    resp = _client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=messages,
    )
    dt_ms = int((time.perf_counter() - t0) * 1000)
    _ai_trace_log([f"AI_TIME call_id=chat_json model={model} duration_ms={dt_ms}"])
    try:
        print(f"[AI] chat_json model={model} duration={dt_ms}ms")
    except Exception:
        pass

    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except Exception:
        # Attempt minimal repair if the model returns non-JSON by mistake
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(raw[start:end + 1])
            else:
                parsed = {}
        except Exception:
            parsed = {}
    # Pretty-print the JSON to console for quick review (truncated)
    try:
        print("[AI JSON chat_json]")
        print(json.dumps(parsed, indent=2)[:2000])
    except Exception:
        pass
    return parsed


def chat_json_system_user(system_prompt: str, user_prompt: str, model: str = "gpt-5-mini") -> Dict[str, Any]:
    """
    JSON-only chat call with an explicit system message and a user message.
    """
    from time import perf_counter
    _ai_trace_log([
        f"AI_CALL call_id=chat_json_system_user model={model} response_format=json_object",
        "SYSTEM=<<<BEGIN",
        * (system_prompt or "").splitlines(),
        "<<<END",
        "USER=<<<BEGIN",
        * (user_prompt or "").splitlines(),
        "<<<END",
    ])
    messages = [
        {"role": "system", "content": system_prompt or ""},
        {"role": "user", "content": user_prompt or ""},
    ]
    t0 = perf_counter()
    resp = _client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=messages,
    )
    dt_ms = int((perf_counter() - t0) * 1000)
    _ai_trace_log([f"AI_TIME call_id=chat_json_system_user model={model} duration_ms={dt_ms}"])
    try:
        print(f"[AI] chat_json_system_user model={model} duration={dt_ms}ms")
    except Exception:
        pass

    raw = resp.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw)
    except Exception:
        try:
            start = raw.find("{"); end = raw.rfind("}")
            parsed = json.loads(raw[start:end+1]) if start != -1 and end != -1 and end > start else {}
        except Exception:
            parsed = {}

    # Pretty-print the JSON to console for quick review (truncated)
    try:
        print("[AI JSON chat_json_system_user]")
        print(json.dumps(parsed, indent=2)[:2000])
    except Exception:
        pass
    return parsed


def extract_text_from_image(image_path: str) -> str:
    """
    Extract visible text (bio, name/age, prompts/answers, interests, location) from a dating profile screenshot.
    """
    prompt = """
    Extract all visible text from this dating profile screenshot.

    Focus on:
    - Profile bio/description text
    - Name and age information
    - Any prompts and answers
    - Interests or hobbies mentioned
    - Location information if visible

    Return only the extracted text content, formatted cleanly without any analysis or commentary.
    """
    # AI trace: log prompt and image path + size (never log base64)
    try:
        _sz = os.path.getsize(image_path)
    except Exception:
        _sz = "?"
    _ai_trace_log([
        "AI_CALL call_id=extract_text_from_image model=gpt-5-mini temperature=0.2",
        "PROMPT=<<<BEGIN",
        *prompt.splitlines(),
        "<<<END",
        f"IMAGE image_path={image_path} image_size={_sz} bytes"
    ])

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64_image(image_path)}"}}
        ]
    }]
    t0 = time.perf_counter()
    resp = _client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
        temperature=0.2
    )
    dt_ms = int((time.perf_counter() - t0) * 1000)
    _ai_trace_log([f"AI_TIME call_id=extract_text_from_image model=gpt-5-mini duration_ms={dt_ms}"])
    try:
        print(f"[AI] extract_text_from_image model=gpt-5-mini duration={dt_ms}ms")
    except Exception:
        pass
    return (resp.choices[0].message.content or "").strip()


def _generate_fallback_flirty_comment(profile_text: str) -> str:
    import random
    profile_lower = (profile_text or "").lower()

    if any(word in profile_lower for word in ['coffee', 'caffeine', 'espresso', 'latte']):
        return "I have a theory that our first coffee date is going to turn into an all-day adventure"
    if any(word in profile_lower for word in ['travel', 'adventure', 'explore', 'wanderlust']):
        return "Your wanderlust is showing - want to explore the city together?"
    if any(word in profile_lower for word in ['food', 'foodie', 'cooking', 'restaurant', 'pizza']):
        return "I'm getting serious 'let's debate the best restaurants over dinner' energy from you"
    if any(word in profile_lower for word in ['music', 'concert', 'festival', 'band']):
        return "Plot twist: what if our music taste is as compatible as I think? Testing required"
    if any(word in profile_lower for word in ['workout', 'gym', 'fitness', 'yoga', 'hike']):
        return "Challenge accepted - but first, let's grab drinks and see if you're as competitive as me"

    flirty_fallbacks = [
        "I'm getting major 'let's grab drinks and see if you're as interesting in person' vibes",
        "Your profile just convinced me we need to test our compatibility over coffee",
        "I have a theory that we'd have amazing chemistry - care to help me test it?",
        "Warning: I'm about to suggest we skip the small talk and go straight to an adventure",
        "Plot twist: what if we actually met up instead of just matching? Wild concept, I know",
        "I'm calling it - we're going to have one of those 'can't believe we met on an app' stories",
        "Fair warning: I'm really good at first dates. Want to find out?",
        "I have a feeling you're trouble in the best way possible. Prove me right?",
        "Your vibe is immaculate - when can we test the in-person chemistry?",
        "I'm convinced we're going to have one of those conversations that goes until 3am",
        "Something tells me you'd be dangerous to take on a date - I'm intrigued"
    ]
    return random.choice(flirty_fallbacks)


def generate_comment(profile_text: str) -> str:
    """
    Generate a flirty, witty dating app comment designed to get a date.
    """
    prompt = f"""
    Based on this dating profile, generate a FLIRTY, WITTY comment that's designed to get a date.

    Profile Content:
    {profile_text}

    STYLE REQUIREMENTS:
    - Be confident and playfully flirty (not aggressive or creepy)
    - Use clever wordplay, puns, or witty observations
    - Reference something specific from their profile to show you actually read it
    - Create intrigue and make them want to respond
    - Suggest meeting up in a clever/indirect way
    - Sound like you're genuinely interested in them as a person
    - Keep it under 40 words for maximum impact
    - Use only plain ASCII characters (no emojis or diacritics)

    AVOID:
    - Generic compliments about looks
    - Boring "hey how are you" openers
    - Overly sexual or inappropriate content
    - Trying too hard to be funny
    - Being too serious or formal
    - Emojis or non-ASCII characters

    Generate ONE flirty, witty comment that will get them excited to meet up:
    """
    text = _chat_text(prompt, temperature=0.7)
    text = text.strip().strip("\"'")
    if not text or len(text) < 10 or text.lower().startswith("hey"):
        return _generate_fallback_flirty_comment(profile_text)
    return text


def generate_contextual_date_comment(profile_analysis: dict, profile_text: str) -> str:
    """
    Generate contextual flirty comments using analysis features like interests, personality, profession, location.
    """
    interests = profile_analysis.get('interests', [])
    personality_traits = profile_analysis.get('personality_traits', [])
    profession = profile_analysis.get('profession', '')
    location = profile_analysis.get('location', '')

    context_info = f"""
    PROFILE ANALYSIS:
    - Interests: {', '.join(interests[:5])}
    - Personality: {', '.join(personality_traits[:3])}
    - Profession: {profession}
    - Location: {location}

    FULL PROFILE TEXT:
    {profile_text[:500]}...
    """

    prompt = f"""
    Create an IRRESISTIBLE, flirty comment that will make them want to meet up ASAP.

    {context_info}

    ADVANCED REQUIREMENTS:
    - Use their specific interests/job/personality to create a unique opener
    - Be confident and slightly cocky (but charming)
    - Create instant chemistry and intrigue
    - Suggest a specific type of date that matches their interests
    - Use humor, wit, or clever observations
    - Maximum 35 words
    - Use only plain ASCII characters (no emojis or diacritics)

    Generate ONE comment that's impossible to ignore:
    """
    text = _chat_text(prompt, temperature=0.7)
    text = text.strip().strip("\"'")
    if not text or len(text) < 15:
        return generate_comment(profile_text)
    return text


def analyze_dating_ui(image_path: str) -> dict:
    """
    Analyze a dating app screenshot and return structured UI analysis JSON.
    """
    prompt = """
    Analyze this dating app screenshot and provide a comprehensive UI analysis in JSON format:

    {
        "has_like_button": true/false,
        "like_button_visible": true/false,
        "profile_quality_score": 1-10,
        "should_like": true/false,
        "reason": "detailed reason for recommendation",
        "ui_elements_detected": ["list", "of", "visible", "elements"],
        "profile_attractiveness": 1-10,
        "text_content_quality": 1-10,
        "conversation_potential": 1-10,
        "red_flags": ["any", "concerning", "elements"],
        "positive_indicators": ["good", "signs", "to", "like"]
    }

    Respond only with valid JSON.
    """
    return _chat_json(prompt, image_path=image_path, temperature=0.0)


def find_ui_elements(image_path: str, element_type: str = "like_button") -> dict:
    """
    Find UI element approximate normalized coordinates and confidence.
    """
    prompt = f"""
    Analyze this dating app screenshot and find the {element_type}.

    Provide precise location in JSON format:
    {{
        "element_found": true/false,
        "approximate_x_percent": 0.0-1.0,
        "approximate_y_percent": 0.0-1.0,
        "confidence": 0.0-1.0,
        "description": "detailed description of what you see",
        "visual_context": "describe surrounding elements",
        "tap_area_size": "small/medium/large"
    }}

    Express coordinates as 0.0..1.0. Respond only with valid JSON.
    """
    return _chat_json(prompt, image_path=image_path, temperature=0.0)


def analyze_profile_scroll_content(image_path: str) -> dict:
    """
    Determine if more content exists below and where to scroll (normalized coordinates).
    """
    prompt = """
    Analyze this dating profile screenshot to determine scrolling needs:

    {
        "has_more_content": true/false,
        "scroll_direction": "up/down/none",
        "content_completion": 0.0-1.0,
        "visible_profile_elements": ["photos", "bio", "prompts", "interests"],
        "should_scroll_down": true/false,
        "scroll_area_center_x": 0.0-1.0,
        "scroll_area_center_y": 0.0-1.0,
        "analysis": "description",
        "scroll_confidence": 0.0-1.0,
        "estimated_content_below": "description"
    }

    Respond only with valid JSON.
    """
    return _chat_json(prompt, image_path=image_path, temperature=0.0)


def get_profile_navigation_strategy(image_path: str) -> dict:
    """
    Recommend navigation action and swipe vectors based on screenshot classification.
    """
    prompt = """
    Analyze this dating app screen to determine navigation strategy:

    {
        "screen_type": "profile/card_stack/other",
        "stuck_indicator": true/false,
        "navigation_action": "swipe_left/swipe_right/scroll_down/tap_next/go_back",
        "swipe_direction": "left/right/up/down",
        "swipe_start_x": 0.0-1.0,
        "swipe_start_y": 0.0-1.0,
        "swipe_end_x": 0.0-1.0,
        "swipe_end_y": 0.0-1.0,
        "confidence": 0.0-1.0,
        "reason": "why this navigation is recommended"
    }

    Respond only with valid JSON.
    """
    return _chat_json(prompt, image_path=image_path, temperature=0.0)


def detect_comment_ui_elements(image_path: str) -> dict:
    """
    Detect comment text field and send button approximate positions.
    """
    prompt = """
    Analyze this dating app comment interface screenshot and find UI elements:

    {
        "comment_field_found": true/false,
        "comment_field_x": 0.0-1.0,
        "comment_field_y": 0.0-1.0,
        "comment_field_confidence": 0.0-1.0,
        "send_button_found": true/false,
        "send_button_x": 0.0-1.0,
        "send_button_y": 0.0-1.0,
        "send_button_confidence": 0.0-1.0,
        "cancel_button_found": true/false,
        "cancel_button_x": 0.0-1.0,
        "cancel_button_y": 0.0-1.0,
        "interface_state": "comment_ready/sending/error/unknown",
        "description": "what you see"
    }

    Respond only with valid JSON.
    """
    return _chat_json(prompt, image_path=image_path, temperature=0.0)


def verify_action_success(image_path: str, action_type: str) -> dict:
    """
    Verify outcomes like like_tap, comment_sent, profile_change using semantic analysis.
    """
    if action_type == "like_tap":
        prompt = """
        Analyze this dating app screenshot to verify if a LIKE action was successful:

        {
            "like_successful": true/false,
            "interface_state": "comment_modal/main_profile/next_profile/error",
            "visible_indicators": ["like_confirmation", "comment_interface", "match_notification"],
            "next_action_available": true/false,
            "confidence": 0.0-1.0,
            "description": "what indicates success or failure"
        }

        Respond only with valid JSON.
        """
    elif action_type == "comment_sent":
        prompt = """
        Analyze this screenshot to verify if a COMMENT was successfully sent:

        {
            "comment_sent": true/false,
            "interface_state": "back_to_profile/match_screen/conversation_started/error",
            "visible_indicators": ["match_notification", "conversation_preview", "success_message"],
            "comment_interface_gone": true/false,
            "confidence": 0.0-1.0,
            "description": "what indicates success"
        }

        Respond only with valid JSON.
        """
    elif action_type == "profile_change":
        prompt = """
        Analyze this screenshot to verify if we successfully moved to a NEW profile:

        {
            "profile_changed": true/false,
            "interface_state": "new_profile/same_profile/loading/error",
            "profile_elements_visible": ["new_photos", "new_name", "new_bio"],
            "stuck_indicator": true/false,
            "confidence": 0.0-1.0,
            "description": "evidence of profile change or not"
        }

        Respond only with valid JSON.
        """
    else:
        prompt = f"""
        Analyze this screenshot for general action verification of type: {action_type}

        {{
            "action_successful": true/false,
            "interface_state": "unknown",
            "confidence": 0.0-1.0,
            "description": "general analysis"
        }}

        Respond only with valid JSON.
        """

    result = _chat_json(prompt, image_path=image_path, temperature=0.0)
    result["verification_type"] = action_type
    return result
