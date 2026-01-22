#!/usr/bin/env python3
"""
Run a single full-scroll scrape + parse using a custom prompt.
Hardwired to Gemini and prints/saves the parsed JSON for easy review.
"""

import json
import os
from datetime import datetime
from textwrap import dedent

import config  # ensure .env is loaded early via config.py

import prompt_engine
import hinge_agent as ha
from agent_config import DEFAULT_CONFIG
from hinge_agent import HingeAgent, HingeAgentState


def build_custom_structured_prompt() -> str:
    return (
        "You are analyzing screenshots from a Hinge dating profile. Extract information strictly from visible text for explicit fields (biometrics boxes like Age, Height, Ethnicity — these are user-provided and trustworthy). For photo descriptions and inferred traits, provide neutral, factual observations from images only — do not guess, judge, moralize, or infer personality/behavior beyond what's visibly present.\n\n"
        "Return exactly one valid JSON object matching this structure, with the same field names, order, and formatting:\n\n"
        "{\n"
        '  "Name": "",\n'
        '  "Gender": "",\n'
        '  "Sexuality": "",\n'
        '  "Age": 0,\n'
        '  "Height": 0,\n'
        '  "Location": "",\n'
        '  "Explicit Ethnicity": "",\n'
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
        '  "Other text on profile not covered by above": "",\n'
        '  "Description of any non-photo media (e.g., poll, voice note)": "",\n'
        '  "Extensive Description of Photo 1": "",\n'
        '  "Extensive Description of Photo 2": "",\n'
        '  "Extensive Description of Photo 3": "",\n'
        '  "Extensive Description of Photo 4": "",\n'
        '  "Extensive Description of Photo 5": "",\n'
        '  "Extensive Description of Photo 6": "",\n'
        '  "Inferred Visual Traits Summary": {\n'
        '    "Apparent Build Category": "",\n'
        '    "Apparent Skin Tone": "",\n'
        '    "Apparent Ethnic Features": "",\n'
        '    "Apparent Hair Features": "",\n'
        '    "Apparent Facial Symmetry and Grooming Level": "",\n'
        '    "Indicators of Fitness or Lifestyle": "",\n'
        '    "Overall Visual Appeal Vibe": "",\n'
        '    "Apparent Age Range Category": "",\n'
        '    "Attire and Style Indicators": "",\n'
        '    "Body Language and Expression": "",\n'
        '    "Visible Enhancements or Features": "",\n'
        '    "Apparent Upper Body Proportions": "",\n'
        '    "Apparent Attractiveness Tier": "",\n'
        '    "Facial Proportion Balance": "",\n'
        '    "Grooming Effort Level": "",\n'
        '    "Presentation Red Flags": "",\n'
        '    "Alternative/Tattoo/Piercing Level": ""\n'
        '  }\n'
        "}\n\n"
        "Rules:\n"
        "- Return only the JSON object, nothing else.\n"
        "- Use exact field names and structure.\n"
        "- For categorical fields, match one of the allowed options exactly or leave empty/null if unclear/not observable.\n"
        "- For free-text fields, keep concise and parseable (comma-separated descriptors when appropriate).\n"
        "- One image may be stitched biometrics — treat as profile text for explicit fields.\n"
        "- Prioritize observable physical traits without exaggeration or sexualization language.\n\n"
        "Extensive Description of Photo X instructions:\n"
        "Provide a detailed, neutral visual summary of the main subject: include clothing style (casual, elegant, sporty, form-fitting, loose, modest, etc.), pose and activity, background elements, facial features (eye shape/color, hair color/texture/length/style, lip shape, nose shape, face shape), skin tone (light, warm tan, olive, medium-brown, deep brown, etc.), body proportions and build (slender, athletic, average, curvy, full-figured, etc.), indicators of grooming (hair, makeup, nails, accessories, glasses, overall neatness), and general presentation (confident, relaxed, approachable, low-key, polished, etc.). Focus only on observable facts.\n\n"
        "Inferred Visual Traits Summary rules:\n"
        "Base solely on photo visuals across all images. Select the single best-fit option for each categorical field. Leave empty if not clearly observable. Use these exact categories:\n\n"
        '"Apparent Build Category": One of: "Very slender/petite", "Slender/lean", "Athletic/toned/fit", "Average/slim", "Average/medium build", "Curvy/hourglass with defined waist", "Curvy/fuller with softer proportions", "Full-figured/plump/voluptuous", "Heavy-set/stocky", "Muscular/built"\n\n'
        '"Apparent Skin Tone": One of: "Very light/pale/fair", "Light/beige", "Warm light/tan", "Olive/medium-tan", "Golden/medium-brown", "Warm brown/deep tan", "Dark-brown/chestnut", "Very dark/ebony/deep"\n\n'
        '"Apparent Ethnic Features": Concise neutral description only if features strongly suggest (e.g., "Southeast Asian traits: warm tan skin, straight dark hair, almond-shaped eyes", "East Asian traits: fair skin, straight black hair, epicanthic folds", "South Asian traits: medium-brown skin, wavy dark hair", "Latina traits: olive skin, curly dark hair, full lips", "Caucasian traits: light skin, varied hair/eye color", "African traits: dark brown skin, curly hair, full lips", "Middle Eastern traits: olive skin, dark hair, prominent nose", "Mixed/ambiguous"). Leave empty if unclear.\n\n'
        '"Apparent Hair Features": Concise e.g., "Long straight dark brown, worn down", "Short wavy blonde bob", "Curly black shoulder-length", "No visible hair"\n\n'
        '"Apparent Facial Symmetry and Grooming Level": One of: "Very high symmetry with polished grooming", "High symmetry with natural/effortless grooming", "High symmetry with casual grooming", "Moderate symmetry with polished grooming", "Moderate symmetry with casual/minimal grooming", "Low symmetry with grooming", "Low symmetry with minimal/no grooming"\n\n'
        '"Indicators of Fitness or Lifestyle": Concise e.g., "Visible muscle tone, athletic poses, sporty wear", "Relaxed poses, no visible tone, casual settings", "Outdoor/active activities, energetic expressions", "Sedentary/lounging poses, no fitness indicators"\n\n'
        '"Overall Visual Appeal Vibe": One of: "Very low-key/simple/understated", "Relaxed/approachable/cute", "Natural/effortless/charming", "Polished/confident/elegant", "High-energy/playful/adventurous", "Edgy/unique/alternative", "Sensual/alluring", "Minimal/basic"\n\n'
        '"Apparent Age Range Category": One of: "Late teens/early 20s (18-22)", "Early-mid 20s (23-26)", "Mid-late 20s (27-29)", "Early 30s (30-33)", "Mid 30s (34-37)", "Late 30s/early 40s (38-42)", "Mid 40s+ (43+)"\n\n'
        '"Attire and Style Indicators": One of: "Very modest/loose/covered", "Casual/comfortable (jeans, t-shirts)", "Low-key/natural (minimal makeup, simple outfits)", "Polished/elegant (dresses, blouses)", "Sporty/active (leggings, athletic wear)", "Form-fitting/suggestive (crop tops, fitted dresses)", "Highly revealing (low-cut, bikinis, bodycon)", "Edgy/alternative (visible tattoos/piercings, bold prints)"\n\n'
        '"Body Language and Expression": One of: "Shy/reserved (averted gaze, closed posture)", "Relaxed/casual (neutral smile, laid-back)", "Approachable/open (direct gaze, soft smile)", "Confident/engaging (direct eye contact, wide smile)", "Playful/flirty (angled poses, subtle smiles)", "Energetic/vibrant (animated, laughing)"\n\n'
        '"Visible Enhancements or Features": Concise list e.g., "Glasses, natural makeup, no piercings/tattoos", "Heavy makeup, multiple piercings, visible tattoos", "Long painted nails, jewelry, contacts", "No notable enhancements"\n\n'
        '"Apparent Upper Body Proportions": One of: "Petite/small/narrow (minimal upper curves)", "Average/balanced/proportional", "Defined/toned (visible collarbones/shoulders)", "Full/curvy (noticeable bust/waist contrast)", "Prominent/voluptuous (emphasized chest area, cleavage visible when applicable)", "Broad/strong (muscular shoulders/upper body)"\n\n'
        '"Apparent Attractiveness Tier": One of: "Low (below average grooming/symmetry/presentation)", "Average", "Above average (good grooming/symmetry/vibe)", "High (very polished/symmetrical/appealing presentation)". Be conservative; require multiple strong positive indicators for "High" (very high symmetry + polished grooming + flattering presentation). Assign "Low" only with significant negatives (low symmetry + minimal grooming + unflattering poses). Default to "Average" if ambiguous.\n\n'
        '"Facial Proportion Balance": One of: "Balanced/proportional", "Slightly unbalanced (minor asymmetry or disproportion)", "Noticeably unbalanced (obvious disproportion in features)"\n\n'
        '"Grooming Effort Level": One of: "Minimal/natural (little to no makeup/styling)", "Moderate/casual (basic makeup/simple styling)", "High/polished (evident makeup/styled hair/accessories)", "Heavy/overdone (thick makeup/multiple layers/excessive styling)"\n\n'
        '"Presentation Red Flags": One of: "None", "Moderate (awkward poses, unflattering angles, poor lighting)", "Significant (visible skin issues, disheveled appearance, extreme disproportion)"\n\n'
        '"Alternative/Tattoo/Piercing Level": One of: "None/minimal (no visible)", "Moderate (1-2 visible piercings or small tattoos)", "High (multiple piercings, large/visible tattoos)"\n'
    )


def _force_gemini_env() -> None:
    os.environ["LLM_PROVIDER"] = "gemini"
    gemini_model = os.getenv("GEMINI_MODEL")
    gemini_small = os.getenv("GEMINI_SMALL_MODEL")
    if gemini_model and not os.getenv("LLM_MODEL"):
        os.environ["LLM_MODEL"] = gemini_model
    if gemini_small and not os.getenv("LLM_SMALL_MODEL"):
        os.environ["LLM_SMALL_MODEL"] = gemini_small


def _disable_eval_llm() -> None:
    # Avoid extra LLM calls; we only want the first scrape/parse prompt.
    ha.evaluate_profile_fields = lambda extracted, model=None: {}


def _init_state(max_profiles: int = 1) -> HingeAgentState:
    return HingeAgentState(
        device=None,
        width=0,
        height=0,
        max_profiles=max_profiles,
        current_profile_index=0,
        profiles_processed=0,
        likes_sent=0,
        comments_sent=0,
        errors_encountered=0,
        stuck_count=0,
        current_screenshot=None,
        profile_text="",
        profile_analysis={},
        decision_reason="",
        previous_profile_text="",
        previous_profile_features={},
        last_action="",
        action_successful=True,
        retry_count=0,
        generated_comment="",
        comment_id="",
        like_button_coords=None,
        like_button_confidence=0.0,
        should_continue=True,
        completion_reason="",
        ai_reasoning="",
        next_tool_suggestion="",
        batch_start_index=0,
        llm_batch_request={},
    )


def main() -> int:
    _force_gemini_env()
    _disable_eval_llm()
    prompt_engine.build_structured_profile_prompt = build_custom_structured_prompt

    cfg = DEFAULT_CONFIG
    cfg.llm_provider = "gemini"
    cfg.extraction_small_model = "small"
    cfg.extraction_model = "large"

    agent = HingeAgent(max_profiles=1, config=cfg)
    state = _init_state(max_profiles=1)

    s = agent.initialize_session_node(state)
    if not s.get("action_successful") or not s.get("should_continue", True):
        print(f"Initialization failed: {s.get('completion_reason', 'Unknown error')}")
        return 1

    s = agent.capture_screenshot_node(s)
    s = agent.analyze_profile_node(s)
    if not s.get("action_successful"):
        print("Analyze profile failed.")
        return 1

    extracted = s.get("extracted_profile") or s.get("profile_analysis") or {}
    llm_meta = (s.get("llm_batch_request", {}) or {}).get("meta", {})
    out = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "llm_provider": os.getenv("LLM_PROVIDER", ""),
            "model": os.getenv("LLM_SMALL_MODEL") or os.getenv("GEMINI_SMALL_MODEL") or "",
            "images_count": llm_meta.get("images_count"),
            "images_paths": llm_meta.get("images_paths", []),
            "timings": s.get("timings", {}),
        },
        "extracted_profile": extracted,
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))

    try:
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("logs", f"prompt_test_{ts}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Wrote results to {out_path}")
    except Exception as e:
        print(f"Failed to write results: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
