#!/usr/bin/env python3
"""
Run a single full-scroll scrape + parse using a custom prompt.
Hardwired to Gemini and prints/saves the parsed JSON for easy review.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List
from textwrap import dedent

import config  # ensure .env is loaded early via config.py

import prompt_engine
import hinge_agent as ha
from agent_config import DEFAULT_CONFIG
from hinge_agent import HingeAgent, HingeAgentState
from llm_client import get_default_model, get_llm_client, resolve_model
from text_utils import normalize_dashes


def LLM1() -> str:
    return (
        "You are analyzing screenshots from a Hinge dating profile. Extract information strictly from visible text for explicit fields (especially the stitched biometrics cards with icons). For photo descriptions and inferred traits, provide neutral, factual observations from images only. Do not guess, judge, moralize, or infer personality or behavior beyond what is explicitly visible in images or written text.\n\n"
        "Return exactly one valid JSON object matching this structure, with the same field names, order, and formatting:\n\n"
        "{\n"
        '  "Core Biometrics (Objective)": {\n'
        '    "Name": "",\n'
        '    "Gender": "",\n'
        '    "Sexuality": "",\n'
        '    "Age": null,\n'
        '    "Height": null,\n'
        '    "Location": "",\n'
        '    "Explicit Ethnicity": "",\n'
        '    "Children": "",\n'
        '    "Family plans": "",\n'
        '    "Covid Vaccine": "",\n'
        '    "Pets": "",\n'
        '    "Zodiac Sign": "",\n'
        '    "Job title": "",\n'
        '    "University": "",\n'
        '    "Religious Beliefs": "",\n'
        '    "Home town": "",\n'
        '    "Politics": "",\n'
        '    "Languages spoken": "",\n'
        '    "Dating Intentions": "",\n'
        '    "Relationship type": "",\n'
        '    "Drinking": "",\n'
        '    "Smoking": "",\n'
        '    "Marijuana": "",\n'
        '    "Drugs": ""\n'
        '  },\n'
        '  "Profile Content (Free Description)": {\n'
        '    "Profile Prompts and Answers": [\n'
        '      {"prompt": "", "answer": ""},\n'
        '      {"prompt": "", "answer": ""},\n'
        '      {"prompt": "", "answer": ""}\n'
        '    ],\n'
        '    "Other text on profile not covered by above": "",\n'
        '    "Description of any non-photo media (e.g., video (identified via timestamp in top right), poll, voice note)": "",\n'
        '    "Extensive Description of Photo 1": "",\n'
        '    "Extensive Description of Photo 2": "",\n'
        '    "Extensive Description of Photo 3": "",\n'
        '    "Extensive Description of Photo 4": "",\n'
        '    "Extensive Description of Photo 5": "",\n'
        '    "Extensive Description of Photo 6": ""\n'
        '  },\n'
        '  "Visual Analysis (Inferred From Images)": {\n'
        '    "Inferred Visual Traits Summary": {\n'
        '      "Face Visibility Quality": "",\n'
        '      "Photo Authenticity / Editing Level": "",\n'
        '      "Apparent Body Fat Level": "",\n'
        '      "Profile Distinctiveness": "",\n'
        '      "Apparent Build Category": "",\n'
        '      "Apparent Skin Tone": "",\n'
        '      "Apparent Ethnic Features": "",\n'
        '      "Hair Color": "",\n'
        '      "Facial Symmetry Level": "",\n'
        '      "Indicators of Fitness or Lifestyle": "",\n'
        '      "Overall Visual Appeal Vibe": "",\n'
        '      "Apparent Age Range Category": "",\n'
        '      "Attire and Style Indicators": "",\n'
        '      "Body Language and Expression": "",\n'
        '      "Visible Enhancements or Features": "",\n'
        '      "Apparent Upper Body Proportions": "",\n'
        '      "Apparent Attractiveness Tier": "",\n'
        '      "Reasoning for attractiveness tier": "",\n'
        '      "Facial Proportion Balance": "",\n'
        '      "Grooming Effort Level": "",\n'
        '      "Presentation Red Flags": "",\n'
        '      "Visible Tattoo Level": "",\n'
        '      "Visible Piercing Level": ""\n'
        '    }\n'
        '  }\n'
        "}\n\n"
        "Rules:\n"
        "- Return only the JSON object, nothing else.\n"
        "- Use exact field names and structure.\n"
        "- Do not add extra keys.\n"
        "- If a value is unclear or not visible, leave it as an empty string \"\" (for string fields) or null (for Age and Height).\n\n"
        "Section rules:\n"
        "1) Core Biometrics (Objective):\n"
        "- Populate only when the value is explicitly visible as profile text in the screenshots, including stitched biometrics cards that use icons.\n"
        "- Do not infer missing values.\n"
        "- Age and Height must be integers when present; otherwise null.\n"
        "- Icon hints: Location uses a pin icon; Home town uses a house icon.\n"
        "- For categorical fields, match one of the allowed options exactly or leave empty if unclear or not visible.\n\n"
        "2) Profile Content (Free Description):\n"
        "- These keys must always exist in the output JSON.\n"
        "- Use short, literal transcriptions or descriptions of what is visibly present (prompts and answers, other visible profile text, and any non-photo media).\n"
        "- If an item is not present, leave the value as an empty string.\n\n"
        "3) Visual Analysis (Inferred From Images):\n"
        "- Base solely on photo visuals.\n"
        "- Do not use any values from Core Biometrics (Objective) to fill or influence Visual Analysis fields.\n"
        "- Treat extraction of Core Biometrics and Visual Analysis as two independent steps; do not force consistency between them.\n"
        "- Disagreement between biometrics and inferred appearance (e.g., age range, ethnic features) is acceptable.\n"
        "- For enum fields, select exactly one allowed option unless the field explicitly allows multiple selections.\n"
        "- Leave fields empty if not clearly observable.\n\n"
        "Allowed options for Core Biometrics (Objective) categorical fields:\n"
        '- "Children": "Don\'t have children", "Have children".\n'
        '- "Family plans": "Don\'t want children", "Want children", "Open to children", "Not sure yet".\n'
        '- "Covid Vaccine": "Vaccinated", "Partially vaccinated", "Not yet vaccinated".\n'
        '- "Dating Intentions": "Life partner", "Long-term relationship", "Long-term relationship, open to short", "Short-term relationship, open to long", "Short term relationship", "Figuring out my dating goals".\n'
        '- "Relationship type": "Monogamy", "Non-Monogamy", "Figuring out my relationship type".\n'
        '- "Drinking": "Yes", "Sometimes", "No".\n'
        '- "Smoking": "Yes", "Sometimes", "No".\n'
        '- "Marijuana": "Yes", "Sometimes", "No".\n'
        '- "Drugs": "Yes", "Sometimes", "No".\n\n'
        "Extensive Description of Photo X instructions:\n"
        "Provide a detailed, neutral visual summary of the main subject: clothing style, pose and activity, background elements, facial features when visible, skin tone, body proportions and build, grooming indicators (hair styling, makeup, nails, accessories, glasses), and overall presentation. Focus only on observable facts.\n\n"
        "Inferred Visual Traits Summary rules and allowed categories:\n\n"
        '"Face Visibility Quality": One of: "Clear face in 3+ photos", "Clear face in 1-2 photos", "Face often partially obscured", "Face mostly not visible"\n\n'
        '"Photo Authenticity / Editing Level": One of: "No obvious filters", "Some filters or mild editing", "Heavy filters/face smoothing", "Unclear"\n\n'
        '"Apparent Body Fat Level": One of: "Low", "Average", "High", "Very high", "Unclear"\n\n'
        '"Profile Distinctiveness": One of: "High (specific/unique)", "Medium", "Low (generic/boilerplate)", "Unclear"\n\n'
        '"Apparent Build Category": One of: "Very slender/petite", "Slender/lean", "Athletic/toned/fit", "Average build", "Curvy (defined waist)", "Curvy (softer proportions)", "Heavy-set/stocky", "Obese/high body fat", "Muscular/built"\n\n'
        '"Apparent Skin Tone": One of: "Very light/pale/fair", "Light/beige", "Warm light/tan", "Olive/medium-tan", "Golden/medium-brown", "Warm brown/deep tan", "Dark-brown/chestnut", "Very dark/ebony/deep"\n\n'
        '"Apparent Ethnic Features": One of: "Ambiguous/unclear", "East Asian-presenting", "Southeast Asian-presenting", "South Asian-presenting", "Indian-presenting", "Jewish/Israeli-presenting", "Arab-presenting", "North African-presenting", "Middle Eastern-presenting (other/unspecified)", "Black/African-presenting", "Latino-presenting", "Nordic/Scandinavian-presenting", "Slavic/Eastern European-presenting", "Mediterranean/Southern European-presenting", "Western/Central European-presenting", "British/Irish-presenting", "White/European-presenting (unspecified)", "Mixed/ambiguous"\n\n'
        '"Hair Color": One of: "Black", "Dark brown", "Medium brown", "Light brown", "Blonde", "Platinum blonde", "Red/ginger", "Gray/white", "Bald/shaved", "Dyed pink", "Dyed blue", "Dyed (unnatural other)", "Dyed (mixed/multiple colors)"\n\n'
        '"Facial Symmetry Level": One of: "Very high", "High", "Moderate", "Low"\n\n'
        '"Indicators of Fitness or Lifestyle": Pick any that apply (comma-separated): "Visible muscle tone", "Athletic poses", "Sporty/athletic clothing", "Outdoor/active settings", "Gym/fitness context visible", "Sedentary/lounging poses", "No visible fitness indicators"\n\n'
        '"Overall Visual Appeal Vibe": One of: "Very low-key/understated", "Natural/effortless", "Polished/elegant", "High-energy/adventurous", "Playful/flirty", "Sensual/alluring", "Edgy/alternative"\n\n'
        '"Apparent Age Range Category": One of: "Late teens/early 20s (18-22)", "Early-mid 20s (23-26)", "Mid-late 20s (27-29)", "Early 30s (30-33)", "Mid 30s (34-37)", "Late 30s/early 40s (38-42)", "Mid 40s+ (43+)"\n\n'
        '"Attire and Style Indicators": One of: "Very modest/covered", "Casual/comfortable", "Low-key/natural", "Polished/elegant", "Sporty/active", "Form-fitting/suggestive", "Highly revealing", "Edgy/alternative"\n\n'
        '"Body Language and Expression": One of: "Shy/reserved", "Relaxed/casual", "Approachable/open", "Confident/engaging", "Playful/flirty", "Energetic/vibrant"\n\n'
        '"Visible Enhancements or Features": Pick any that apply (comma-separated): "None visible", "Glasses", "Sunglasses", "Makeup (light)", "Makeup (heavy)", "Jewelry", "Painted nails", "Very long nails (2cm+)", "Hair extensions/wig (obvious)", "False eyelashes (obvious)", "Hat/cap/beanie (worn in most photos)"\n\n'
        '"Apparent Upper Body Proportions": One of: "Petite/small/narrow", "Average/balanced/proportional", "Defined/toned", "Full/curvy", "Prominent/voluptuous", "Broad/strong"\n\n'
        '"Apparent Attractiveness Tier": One of: "Very unattractive/morbidly obese", "Low", "Average", "Above average", "High", "Very attractive", "Extremely attractive", "Supermodel"\n\n'
        '"Apparent Attractiveness Tier (CRITICAL - CONSERVATIVE SCORING REQUIRED)": Default to "Average". This is a mainstream/common-judgment rating. If uncertain between two tiers, ALWAYS choose the LOWER tier. If "Face Visibility Quality" is NOT "Clear face in 3+ photos", do NOT output tiers ABOVE "Above average". For tiers ABOVE "Average", require clear evidence across multiple photos; do not promote based on vibes, style, or distinctiveness alone. If multiple clear limiting factors are present, do not avoid using "Low" or "Very unattractive/morbidly obese". Use "Supermodel" only in extremely rare cases where the subject clearly stands out far above typical dating-app profiles across multiple photos (exceptional facial symmetry, consistently flattering angles/lighting, and polished grooming).\n\n'
        '"Reasoning for attractiveness tier": Provide ONE concise sentence listing observable factors. Requirements by tier: if tier is ABOVE "Average", include at least 1 clear positive AND 1 clear limiting factor; if tier is "Average", include at least 1 positive AND 1 limiting factor; if tier is "Low" or "Very unattractive/morbidly obese", include 2 or more limiting factors (positives optional). Avoid vague words.\n\n'
        '"Facial Symmetry Level (CONSERVATIVE)": Default to "Moderate". Use "High" or "Very high" only when symmetry is clearly and consistently strong across multiple photos. Do NOT infer symmetry from grooming or styling.\n\n'
        '"Apparent Age Range Category (IMPORTANT - VISUAL ONLY)": This field must be based ONLY on facial appearance in the photos. Do NOT reconcile or align with the stated age. Disagreement is expected and desirable. If uncertain between adjacent age ranges, ALWAYS choose the OLDER range. Overestimating apparent age is preferred to underestimating it.\n\n'
        '"Facial Proportion Balance": One of: "Balanced/proportional", "Slightly unbalanced", "Noticeably unbalanced"\n\n'
        '"Grooming Effort Level": One of: "Minimal/natural", "Moderate/casual", "High/polished", "Heavy/overdone"\n\n'
        '"Presentation Red Flags": Pick any that apply (comma-separated): "None", "Poor lighting", "Blurry/low resolution", "Unflattering angle", "Heavy filters/face smoothing", "Sunglasses in most photos", "Face mostly obscured", "Group photos unclear who is the subject", "Too many distant shots", "Mirror selfie cluttered", "Messy background", "Only one clear solo photo", "Awkward cropping", "Overexposed/washed out", "Inconsistent appearance across photos", "Entitlement language in prompts", "Transactional dating expectations stated explicitly", "Rigid gender role expectations stated explicitly", "Aggressive negativity in prompts", "Excessive rules or dealbreakers listed", "Hostile or contemptuous humor in prompts", "Passive-aggressive tone in prompts", "Explicit materialism or status demands"\n\n'
        '"Visible Tattoo Level": One of: "None visible", "Small/minimal", "Moderate", "High"\n\n'
        '"Visible Piercing Level": One of: "None visible", "Minimal", "Moderate", "High"\n'
        'Rule (piercings): "Minimal" = ONLY standard single earlobe earrings. ANY septum/nose/eyebrow/lip or multiple ear piercings MUST be "Moderate" or "High". If uncertain between adjacent levels, choose the HIGHER level.\n\n'
    )






def LLM2(home_town: str, job_title: str, university: str) -> str:
    """
    Build the enrichment prompt for evaluating Home town, Job title, University.
    Returns a single prompt string instructing the model to output EXACTLY one JSON object.
    This version is aligned to the test harness (no legacy modifiers).
    """
    parts = [
        "You are enriching structured dating profile fields for a scoring system. Use ONLY the provided text. Do not browse. Be conservative when uncertain, but apply a slight optimistic bias when inferring future earning potential.\n\n",
        "INPUT FIELDS (from the extracted JSON):\n",
        '- "Home town" (string; may be city/region/country or empty)\n',
        '- "Job title" (string; may be empty)\n',
        '- "University" (string; may be empty)\n\n',
        "VALUES:\n",
        f'Home town: "{home_town or ""}"\n',
        f'Job title: "{job_title or ""}"\n',
        f'University: "{university or ""}"\n\n',
        "YOUR TASKS (3):\n",
        '1) Resolve "Home town" to an ISO 3166-1 alpha-2 country code (uppercase). If it is a UK city/area (e.g., "Wembley", "Harrow", "Manchester"), return "GB".\n',
        '2) Estimate FUTURE EARNING POTENTIAL (TIER) from the vague job/study field AND the university context. Titles are often minimal (e.g., "Tech", "Finance", "Product", "Student", "PhD"). Use the tier table in section B and return the corresponding band "T0"-"T4". When uncertain between two adjacent tiers, be slightly optimistic and choose the higher tier by at most one step.\n',
        '3) Check if "University" matches an elite list (case-insensitive), and return a 1/0 flag and the matched canonical name.\n\n',
        "--------------------------------------------------------------------------------\n",
        "A) home_country_iso\n",
        '- If unresolved: home_country_iso = "" and home_country_confidence = 0.0.\n\n',
        "B) FUTURE EARNING POTENTIAL (tiers T0-T4) -> job.band\n",
        "- Goal: infer likely earning trajectory within ~10 years using BOTH job/study field and university context (if visible). Classify into one of these tiers:\n",
        "  T0: Low/no trajectory. Clear low-mobility sectors with low ceiling and no elite cues: retail, hospitality, customer service, basic admin, charity/NGO support, nanny/TA, generic creative with no domain anchor.\n",
        '  T1: Stable but capped. Teacher, nurse, social worker, marketing/HR/recruitment/ops/comms, public-sector researcher, therapist/psychology, non-STEM PhD, generic "research".\n',
        '  T2: Mid/high potential. Engineer, analyst, product manager, consultant, doctor, solicitor, finance, data, law, scientist, sales, generic "tech/software/PM", STEM PhD, or STUDENT with elite STEM context.\n',
        '  T3: High trajectory. Investment/banking, management consulting, quant, strategy, PE/VC, corporate law (Magic Circle), AI/data scientist, Big-Tech-calibre product/engineering, "Head/Lead/Director" (early leadership cues).\n',
        "  T4: Exceptional (rare). Partner/Principal/Director (large firm), VP, funded founder with staff, senior specialist physicians, staff/principal engineer. Require strong textual cues.\n",
        '- Beneficial-doubt rule for missing or humorous titles: If the job field is empty or clearly humorous (e.g., "Glorified babysitter"), assign **T1 by default**, and upgrade to **T2** if elite-STEM education or strong sector hints justify it.\n',
        "- University influence: If elite uni AND STEM/quant field hints, allow T2-T3 even for \"Student/PhD\". If elite uni but non-STEM, at most T1-T2 unless sector hints justify higher.\n",
        "- Vague sector keyword mapping (examples, not exhaustive):\n",
        '   "Tech/Software/Engineer/Data/PM/AI" -> T2; consider T3 with elite context.\n',
        '   "Finance/Banking/Investment/Analyst" -> T2; consider T3 with elite context.\n',
        '   "Consulting/Strategy" -> T2; consider T3 with elite context.\n',
        '   "Law/Solicitor/Legal" -> T2; consider T3 with Magic Circle/elite context.\n',
        '   "Marketing/HR/Recruitment/Ops/Comms/Education/Therapy/Research" -> T1 by default; upgrade to T2 only with strong signals.\n',
        "- Confidence: return confidence 0.0-1.0 for the chosen tier. Do NOT downscale the tier due to low confidence; the optimism rule already limits to a one-step upgrade.\n\n",
        "C) university_elite\n",
        "- Elite universities list (case-insensitive exact name match after trimming):\n",
        '  ["University of Oxford","University of Cambridge","Imperial College London", "UCL", "London School of Economics","Harvard University","Yale University","Princeton University","Stanford University","MIT","Columbia University","ETH Zurich","EPFL","University of Copenhagen","Sorbonne University","University of Tokyo","National University of Singapore","Tsinghua University","Peking University","University of Toronto","Australian National University","University of Melbourne","University of Hong Kong"]\n',
        '- Matching rule: If the University field contains multiple names or partial mentions (e.g., "Oxford, PhD @ UCL"), treat it as elite if ANY part contains an elite name (case-insensitive). Set matched_university_name to the canonical elite name.\n',
        "- university_elite = 1 if matched, else 0\n",
        '- matched_university_name = the canonical elite name matched, else "".\n\n',
        "--------------------------------------------------------------------------------\n",
        "OUTPUT EXACTLY ONE JSON OBJECT (no commentary, no code fences):\n\n",
        "{\n",
        '  "home_country_iso": "",           // ISO alpha-2 or ""\n',
        '  "home_country_confidence": 0.0,   // 0.0-1.0\n\n',
        '  "job": {\n',
        '    "normalized_title": "",         // concise title or "Unknown"\n',
        '    "band": "",                     // USE "T0"|"T1"|"T2"|"T3"|"T4"\n',
        '    "confidence": 0.0,              // 0.0-1.0 confidence for the chosen tier\n',
        '    "band_reason": ""               // one short sentence justifying the tier choice\n',
        "  },\n\n",
        '  "university_elite": 0,            // 1 or 0\n',
        '  "matched_university_name": ""\n',
        "}\n",
    ]
    return "".join(parts)


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


def _get_core(extracted: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(extracted, dict):
        return {}
    core = extracted.get("Core Biometrics (Objective)", {})
    return core if isinstance(core, dict) else {}


def _get_visual(extracted: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(extracted, dict):
        return {}
    visual_root = extracted.get("Visual Analysis (Inferred From Images)", {})
    if not isinstance(visual_root, dict):
        return {}
    traits = visual_root.get("Inferred Visual Traits Summary", {})
    return traits if isinstance(traits, dict) else {}


def _norm_value(value: Any) -> str:
    if value is None:
        return ""
    s = normalize_dashes(str(value)).strip().lower()
    return " ".join(s.split())


def _split_csv(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    if not s:
        return []
    return [part.strip() for part in s.split(",") if part.strip()]


def run_profile_eval_llm(extracted: Dict[str, Any], model: str | None = None) -> Dict[str, Any]:
    core = _get_core(extracted)
    home_town = core.get("Home town", "") or ""
    job_title = core.get("Job title", "") or ""
    university = core.get("University", "") or ""

    prompt = LLM2(home_town, job_title, university)
    requested_model = model or get_default_model()
    resolved_model = resolve_model(requested_model)

    try:
        t0 = time.perf_counter()
        resp = get_llm_client().chat.completions.create(
            model=resolved_model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        _ = int((time.perf_counter() - t0) * 1000)
        raw = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            parsed = _default_profile_eval()
        parsed = normalize_dashes(parsed)
        return parsed
    except Exception:
        return _default_profile_eval()


# ---- Scoring rules (v0) ----
HOME_COUNTRY_PLUS2 = {"NO", "SE", "DK", "FI", "IS", "EE", "LV", "LT", "UA", "RU", "BY"}
HOME_COUNTRY_PLUS1 = {"IE", "DE", "FR", "NL", "BE", "LU", "CH", "AT", "IT", "ES", "PT", "PL", "CZ", "CA", "US", "AU", "NZ", "JP", "KR", "SG", "IL", "AE"}
HOME_COUNTRY_MINUS1 = {"AR", "BR", "CL", "CO", "PE", "EC", "UY", "PY", "BO", "VE", "GY", "SR", "ID", "MY", "TH", "VN", "PH", "KH", "LA", "MM", "BN"}

AGE_RANGE_MIDPOINTS = {
    "Late teens/early 20s (18-22)": 20.0,
    "Early-mid 20s (23-26)": 24.5,
    "Mid-late 20s (27-29)": 28.0,
    "Early 30s (30-33)": 31.5,
    "Mid 30s (34-37)": 35.5,
    "Late 30s/early 40s (38-42)": 40.0,
    "Mid 40s+ (43+)": 45.0,
}
AGE_RANGE_BOUNDS = {
    "Late teens/early 20s (18-22)": (18, 22),
    "Early-mid 20s (23-26)": (23, 26),
    "Mid-late 20s (27-29)": (27, 29),
    "Early 30s (30-33)": (30, 33),
    "Mid 30s (34-37)": (34, 37),
    "Late 30s/early 40s (38-42)": (38, 42),
    "Mid 40s+ (43+)": (43, None),
}


def _score_profile_long(extracted: Dict[str, Any], eval_result: Dict[str, Any]) -> Dict[str, Any]:
    core = _get_core(extracted)
    visual = _get_visual(extracted)

    contribs: List[Dict[str, Any]] = []
    hard_kills: List[Dict[str, Any]] = []

    def record(section: str, field: str, value: Any, delta: int) -> None:
        if delta == 0:
            return
        entry = {"section": section, "field": field, "value": value, "delta": int(delta)}
        contribs.append(entry)
        if delta <= -1000:
            hard_kills.append(entry)

    def core_val(key: str) -> Any:
        return core.get(key, "")

    def visual_val(key: str) -> Any:
        return visual.get(key, "")

    # Core Biometrics
    gender = core_val("Gender")
    gender_norm = _norm_value(gender)
    if gender_norm == _norm_value("Non-binary"):
        record("Core Biometrics", "Gender", gender, -1000)

    children = core_val("Children")
    if _norm_value(children) == _norm_value("Have children"):
        record("Core Biometrics", "Children", children, -1000)

    covid = core_val("Covid Vaccine")
    if str(covid).strip():
        record("Core Biometrics", "Covid Vaccine", covid, -5)

    dating = core_val("Dating Intentions")
    dating_norm = _norm_value(dating)
    if dating_norm == _norm_value("Life partner"):
        record("Core Biometrics", "Dating Intentions", dating, -5)

    drinking = core_val("Drinking")
    if _norm_value(drinking) == _norm_value("Sometimes"):
        record("Core Biometrics", "Drinking", drinking, +5)

    smoking = core_val("Smoking")
    smoking_norm = _norm_value(smoking)
    if smoking_norm == _norm_value("Yes"):
        record("Core Biometrics", "Smoking", smoking, -1000)
    elif smoking_norm == _norm_value("Sometimes"):
        record("Core Biometrics", "Smoking", smoking, -20)

    marijuana = core_val("Marijuana")
    marijuana_norm = _norm_value(marijuana)
    if marijuana_norm == _norm_value("Yes"):
        record("Core Biometrics", "Marijuana", marijuana, -1000)
    elif marijuana_norm == _norm_value("Sometimes"):
        record("Core Biometrics", "Marijuana", marijuana, -20)

    drugs = core_val("Drugs")
    drugs_norm = _norm_value(drugs)
    if drugs_norm == _norm_value("Yes"):
        record("Core Biometrics", "Drugs", drugs, -1000)
    elif drugs_norm == _norm_value("Sometimes"):
        record("Core Biometrics", "Drugs", drugs, -20)

    sexuality = core_val("Sexuality")
    sexuality_norm = _norm_value(sexuality)
    if sexuality_norm == _norm_value("Bisexual"):
        record("Core Biometrics", "Sexuality", sexuality, +5)
    elif sexuality_norm and sexuality_norm != _norm_value("Straight"):
        record("Core Biometrics", "Sexuality", sexuality, -5)

    zodiac = core_val("Zodiac Sign")
    if str(zodiac).strip():
        record("Core Biometrics", "Zodiac Sign", zodiac, -5)

    religion = core_val("Religious Beliefs")
    religion_norm = _norm_value(religion)
    if religion_norm == _norm_value("Atheist"):
        record("Core Biometrics", "Religious Beliefs", religion, +10)
    elif religion_norm == _norm_value("Jewish"):
        record("Core Biometrics", "Religious Beliefs", religion, +10)
    elif religion_norm == _norm_value("Muslim"):
        record("Core Biometrics", "Religious Beliefs", religion, -1000)
    elif religion_norm:
        record("Core Biometrics", "Religious Beliefs", religion, -10)

    politics = core_val("Politics")
    politics_norm = _norm_value(politics)
    if politics_norm == _norm_value("Conservative"):
        record("Core Biometrics", "Politics", politics, +5)
    elif politics_norm == _norm_value("Not political"):
        record("Core Biometrics", "Politics", politics, +5)
    elif politics_norm == _norm_value("Liberal"):
        record("Core Biometrics", "Politics", politics, -5)

    # Age weighting (declared age only)
    declared_age = core_val("Age")
    declared_age_int = None
    try:
        declared_age_int = int(declared_age) if declared_age is not None and str(declared_age).strip() != "" else None
    except Exception:
        declared_age_int = None
    if declared_age_int is not None:
        if 18 <= declared_age_int <= 21:
            record("Core Biometrics", "Age", "18-21", +20)
        elif 22 <= declared_age_int <= 24:
            record("Core Biometrics", "Age", "22-24", +30)
        elif 25 <= declared_age_int <= 27:
            record("Core Biometrics", "Age", "25-27", +10)
        elif 28 <= declared_age_int <= 30:
            record("Core Biometrics", "Age", "28-30", 0)
        elif 31 <= declared_age_int <= 35:
            record("Core Biometrics", "Age", "31-35", -10)
        elif 36 <= declared_age_int <= 40:
            record("Core Biometrics", "Age", "36-40", -20)

    # Height weighting (declared height only)
    height = core_val("Height")
    height_int = None
    try:
        height_int = int(height) if height is not None and str(height).strip() != "" else None
    except Exception:
        height_int = None
    if height_int is not None:
        if height_int >= 185:
            record("Core Biometrics", "Height", f"{height_int}", +20)
        elif height_int > 175:
            record("Core Biometrics", "Height", f"{height_int}", +10)

    # Visual Analysis
    face_visibility = visual_val("Face Visibility Quality")
    face_visibility_norm = _norm_value(face_visibility)
    face_visibility_deltas = {
        _norm_value("Clear face in 3+ photos"): 0,
        _norm_value("Clear face in 1-2 photos"): -5,
        _norm_value("Face often partially obscured"): -10,
        _norm_value("Face mostly not visible"): -20,
    }
    if face_visibility_norm in face_visibility_deltas:
        record(
            "Visual Analysis",
            "Face Visibility Quality",
            face_visibility,
            face_visibility_deltas[face_visibility_norm],
        )

    photo_editing = visual_val("Photo Authenticity / Editing Level")
    photo_editing_norm = _norm_value(photo_editing)
    photo_editing_deltas = {
        _norm_value("No obvious filters"): 0,
        _norm_value("Some filters or mild editing"): -5,
        _norm_value("Heavy filters/face smoothing"): -20,
        _norm_value("Unclear"): 0,
    }
    if photo_editing_norm in photo_editing_deltas:
        record(
            "Visual Analysis",
            "Photo Authenticity / Editing Level",
            photo_editing,
            photo_editing_deltas[photo_editing_norm],
        )

    body_fat = visual_val("Apparent Body Fat Level")
    body_fat_norm = _norm_value(body_fat)
    body_fat_deltas = {
        _norm_value("Low"): 0,
        _norm_value("Average"): 0,
        _norm_value("High"): -10,
        _norm_value("Very high"): -1000,
        _norm_value("Unclear"): 0,
    }
    if body_fat_norm in body_fat_deltas:
        record(
            "Visual Analysis",
            "Apparent Body Fat Level",
            body_fat,
            body_fat_deltas[body_fat_norm],
        )

    distinctiveness = visual_val("Profile Distinctiveness")
    distinctiveness_norm = _norm_value(distinctiveness)
    distinctiveness_deltas = {
        _norm_value("High (specific/unique)"): 5,
        _norm_value("Medium"): 0,
        _norm_value("Low (generic/boilerplate)"): -5,
        _norm_value("Unclear"): 0,
    }
    if distinctiveness_norm in distinctiveness_deltas:
        record(
            "Visual Analysis",
            "Profile Distinctiveness",
            distinctiveness,
            distinctiveness_deltas[distinctiveness_norm],
        )

    attractiveness = visual_val("Apparent Attractiveness Tier")
    attractiveness_norm = _norm_value(attractiveness)
    attractiveness_deltas = {
        _norm_value("Very unattractive/morbidly obese"): -1000,
        _norm_value("Low"): -1000,
        _norm_value("Average"): -20,
        _norm_value("Above average"): 5,
        _norm_value("High"): 10,
        _norm_value("Very attractive"): 20,
        _norm_value("Extremely attractive"): 30,
        _norm_value("Supermodel"): 40,
    }
    if attractiveness_norm in attractiveness_deltas:
        att_delta = attractiveness_deltas[attractiveness_norm]
        if face_visibility_norm != _norm_value("Clear face in 3+ photos"):
            att_delta = min(att_delta, 5)
        if photo_editing_norm == _norm_value("Heavy filters/face smoothing"):
            att_delta = min(att_delta, 0)
        record(
            "Visual Analysis",
            "Apparent Attractiveness Tier",
            attractiveness,
            att_delta,
        )

    symmetry = visual_val("Facial Symmetry Level")
    symmetry_norm = _norm_value(symmetry)
    if symmetry_norm == _norm_value("Low"):
        record("Visual Analysis", "Facial Symmetry Level", symmetry, -1000)
    elif symmetry_norm == _norm_value("Moderate"):
        record("Visual Analysis", "Facial Symmetry Level", symmetry, -20)

    hair_color = visual_val("Hair Color")
    if _norm_value(hair_color) == _norm_value("Red/ginger"):
        record("Visual Analysis", "Hair Color", hair_color, +10)

    tattoo = visual_val("Visible Tattoo Level")
    if _norm_value(tattoo) == _norm_value("High"):
        record("Visual Analysis", "Visible Tattoo Level", tattoo, -10)

    piercing = visual_val("Visible Piercing Level")
    piercing_norm = _norm_value(piercing)
    if piercing_norm == _norm_value("High"):
        record("Visual Analysis", "Visible Piercing Level", piercing, -1000)
    elif piercing_norm == _norm_value("Moderate"):
        record("Visual Analysis", "Visible Piercing Level", piercing, -20)
    elif piercing_norm == _norm_value("None visible"):
        record("Visual Analysis", "Visible Piercing Level", piercing, +5)

    build = visual_val("Apparent Build Category")
    build_norm = _norm_value(build)
    if build_norm == _norm_value("Obese/high body fat"):
        record("Visual Analysis", "Apparent Build Category", build, -1000)
    elif build_norm == _norm_value("Curvy (softer proportions)"):
        record("Visual Analysis", "Apparent Build Category", build, -10)
    elif build_norm == _norm_value("Muscular/built"):
        record("Visual Analysis", "Apparent Build Category", build, +10)

    skin = visual_val("Apparent Skin Tone")
    skin_norm = _norm_value(skin)
    if skin_norm in {_norm_value("Golden/medium-brown"), _norm_value("Warm brown/deep tan")}:
        record("Visual Analysis", "Apparent Skin Tone", skin, -20)
    elif skin_norm in {_norm_value("Dark-brown/chestnut"), _norm_value("Very dark/ebony/deep")}:
        record("Visual Analysis", "Apparent Skin Tone", skin, -1000)

    ethnic = visual_val("Apparent Ethnic Features")
    ethnic_norm = _norm_value(ethnic)
    if ethnic_norm == _norm_value("Southeast Asian-presenting"):
        record("Visual Analysis", "Apparent Ethnic Features", ethnic, -20)
    elif ethnic_norm in {
        _norm_value("Nordic/Scandinavian-presenting"),
        _norm_value("Slavic/Eastern European-presenting"),
    }:
        record("Visual Analysis", "Apparent Ethnic Features", ethnic, +5)

    upper_body = visual_val("Apparent Upper Body Proportions")
    upper_norm = _norm_value(upper_body)
    if upper_norm == _norm_value("Petite/small/narrow"):
        record("Visual Analysis", "Apparent Upper Body Proportions", upper_body, -5)
    elif upper_norm and upper_norm != _norm_value("Average/balanced/proportional"):
        record("Visual Analysis", "Apparent Upper Body Proportions", upper_body, +5)

    enhancements = _split_csv(visual_val("Visible Enhancements or Features"))
    for item in enhancements:
        item_norm = _norm_value(item)
        if item_norm == _norm_value("Glasses"):
            record("Visual Analysis", "Visible Enhancements or Features", item, +5)
        elif item_norm == _norm_value("Makeup (heavy)"):
            record("Visual Analysis", "Visible Enhancements or Features", item, -10)
        elif item_norm == _norm_value("Very long nails (2cm+)"):
            record("Visual Analysis", "Visible Enhancements or Features", item, -10)
        elif item_norm == _norm_value("False eyelashes (obvious)"):
            record("Visual Analysis", "Visible Enhancements or Features", item, -5)

    red_flags = _split_csv(visual_val("Presentation Red Flags"))
    for flag in red_flags:
        flag_norm = _norm_value(flag)
        if flag_norm == _norm_value("None"):
            continue
        if flag_norm == _norm_value("Heavy filters/face smoothing") and photo_editing_norm == _norm_value("Heavy filters/face smoothing"):
            continue
        record("Visual Analysis", "Presentation Red Flags", flag, -5)

    # Profile Evaluation (LLM2)
    job_band = _norm_value((eval_result.get("job") or {}).get("band", ""))
    if job_band == _norm_value("T0"):
        record("Profile Eval", "Job Tier", "T0", -20)
    elif job_band == _norm_value("T1"):
        record("Profile Eval", "Job Tier", "T1", -10)
    elif job_band == _norm_value("T3"):
        record("Profile Eval", "Job Tier", "T3", +10)
    elif job_band == _norm_value("T4"):
        record("Profile Eval", "Job Tier", "T4", +20)

    university_elite = int(eval_result.get("university_elite", 0) or 0)
    if university_elite == 1:
        record("Profile Eval", "University Elite", "Yes", +10)

    home_iso = str(eval_result.get("home_country_iso", "") or "").upper().strip()
    home_score = 0
    if home_iso == "US":
        home_score = 20
    elif home_iso in HOME_COUNTRY_PLUS2:
        home_score = 10
    elif home_iso in HOME_COUNTRY_PLUS1:
        home_score = 5
    elif home_iso in HOME_COUNTRY_MINUS1:
        home_score = -10
    if home_score:
        record("Profile Eval", "Home Country", home_iso or "(unresolved)", home_score)

    # Age delta (logged only)
    apparent_age = visual_val("Apparent Age Range Category")
    apparent_mid = AGE_RANGE_MIDPOINTS.get(apparent_age) if isinstance(apparent_age, str) else None
    apparent_bounds = AGE_RANGE_BOUNDS.get(apparent_age) if isinstance(apparent_age, str) else None
    age_delta = None
    if declared_age_int is not None and apparent_mid is not None:
        in_range = False
        if apparent_bounds is not None:
            low, high = apparent_bounds
            if high is None:
                in_range = declared_age_int >= low
            else:
                in_range = low <= declared_age_int <= high
        age_delta = 0.0 if in_range else round(float(apparent_mid) - float(declared_age_int), 2)

    score_total = sum(c["delta"] for c in contribs)

    return {
        "score": int(score_total),
        "hard_kills": hard_kills,
        "contributions": contribs,
        "signals": {
            "declared_age": declared_age_int,
            "apparent_age_range": apparent_age if isinstance(apparent_age, str) else "",
            "apparent_age_range_midpoint": apparent_mid,
            "age_delta": age_delta,
        },
        "profile_eval_inputs": {
            "job_band": (eval_result.get("job") or {}).get("band", ""),
            "university_elite": university_elite,
            "home_country_iso": home_iso,
        },
    }


def _score_profile_short(extracted: Dict[str, Any], eval_result: Dict[str, Any]) -> Dict[str, Any]:
    core = _get_core(extracted)
    visual = _get_visual(extracted)

    contribs: List[Dict[str, Any]] = []
    hard_kills: List[Dict[str, Any]] = []

    def record(section: str, field: str, value: Any, delta: int) -> None:
        if delta == 0:
            return
        entry = {"section": section, "field": field, "value": value, "delta": int(delta)}
        contribs.append(entry)
        if delta <= -1000:
            hard_kills.append(entry)

    def core_val(key: str) -> Any:
        return core.get(key, "")

    def visual_val(key: str) -> Any:
        return visual.get(key, "")

    # Core Biometrics
    gender = core_val("Gender")
    gender_norm = _norm_value(gender)
    if gender_norm == _norm_value("Non-binary"):
        record("Core Biometrics", "Gender", gender, -1000)

    children = core_val("Children")
    if _norm_value(children) == _norm_value("Have children"):
        record("Core Biometrics", "Children", children, -20)

    covid = core_val("Covid Vaccine")
    if str(covid).strip():
        record("Core Biometrics", "Covid Vaccine", covid, -5)

    dating = core_val("Dating Intentions")
    dating_norm = _norm_value(dating)
    if dating_norm == _norm_value("Life partner"):
        record("Core Biometrics", "Dating Intentions", dating, -5)
    elif dating_norm in {
        _norm_value("Short term relationship"),
        _norm_value("Short-term relationship, open to long"),
        _norm_value("Figuring out my dating goals"),
    }:
        record("Core Biometrics", "Dating Intentions", dating, +10)

    relationship = core_val("Relationship type")
    relationship_norm = _norm_value(relationship)
    if relationship_norm in {
        _norm_value("Non-Monogamy"),
        _norm_value("Figuring out my relationship type"),
    }:
        record("Core Biometrics", "Relationship type", relationship, +10)

    drinking = core_val("Drinking")
    if _norm_value(drinking) == _norm_value("Sometimes"):
        record("Core Biometrics", "Drinking", drinking, +5)

    smoking = core_val("Smoking")
    smoking_norm = _norm_value(smoking)
    if smoking_norm == _norm_value("Yes"):
        record("Core Biometrics", "Smoking", smoking, -1000)
    elif smoking_norm == _norm_value("Sometimes"):
        record("Core Biometrics", "Smoking", smoking, -20)

    marijuana = core_val("Marijuana")
    marijuana_norm = _norm_value(marijuana)
    if marijuana_norm == _norm_value("Yes"):
        record("Core Biometrics", "Marijuana", marijuana, -20)
    elif marijuana_norm == _norm_value("Sometimes"):
        record("Core Biometrics", "Marijuana", marijuana, -20)

    drugs = core_val("Drugs")
    drugs_norm = _norm_value(drugs)
    if drugs_norm == _norm_value("Yes"):
        record("Core Biometrics", "Drugs", drugs, -1000)
    elif drugs_norm == _norm_value("Sometimes"):
        record("Core Biometrics", "Drugs", drugs, -20)

    sexuality = core_val("Sexuality")
    sexuality_norm = _norm_value(sexuality)
    if sexuality_norm == _norm_value("Bisexual"):
        record("Core Biometrics", "Sexuality", sexuality, +5)
    elif sexuality_norm and sexuality_norm != _norm_value("Straight"):
        record("Core Biometrics", "Sexuality", sexuality, -5)

    zodiac = core_val("Zodiac Sign")
    if str(zodiac).strip():
        record("Core Biometrics", "Zodiac Sign", zodiac, -5)

    religion = core_val("Religious Beliefs")
    religion_norm = _norm_value(religion)
    if religion_norm == _norm_value("Atheist"):
        record("Core Biometrics", "Religious Beliefs", religion, +10)
    elif religion_norm == _norm_value("Jewish"):
        record("Core Biometrics", "Religious Beliefs", religion, +10)
    elif religion_norm == _norm_value("Muslim"):
        record("Core Biometrics", "Religious Beliefs", religion, -1000)
    elif religion_norm:
        record("Core Biometrics", "Religious Beliefs", religion, -10)

    politics = core_val("Politics")
    politics_norm = _norm_value(politics)
    if politics_norm == _norm_value("Conservative"):
        record("Core Biometrics", "Politics", politics, +5)
    elif politics_norm == _norm_value("Not political"):
        record("Core Biometrics", "Politics", politics, +5)
    elif politics_norm == _norm_value("Liberal"):
        record("Core Biometrics", "Politics", politics, -5)

    # Age weighting (declared age only)
    declared_age = core_val("Age")
    declared_age_int = None
    try:
        declared_age_int = int(declared_age) if declared_age is not None and str(declared_age).strip() != "" else None
    except Exception:
        declared_age_int = None
    if declared_age_int is not None:
        if 36 <= declared_age_int <= 40:
            record("Core Biometrics", "Age", "36-40", -10)
        elif declared_age_int >= 41:
            record("Core Biometrics", "Age", "41+", -20)

    # Height weighting (declared height only)
    height = core_val("Height")
    height_int = None
    try:
        height_int = int(height) if height is not None and str(height).strip() != "" else None
    except Exception:
        height_int = None
    if height_int is not None:
        if height_int >= 185:
            record("Core Biometrics", "Height", f"{height_int}", +20)
        elif height_int > 175:
            record("Core Biometrics", "Height", f"{height_int}", +10)

    # Visual Analysis
    face_visibility = visual_val("Face Visibility Quality")
    face_visibility_norm = _norm_value(face_visibility)
    face_visibility_deltas = {
        _norm_value("Clear face in 3+ photos"): 0,
        _norm_value("Clear face in 1-2 photos"): -5,
        _norm_value("Face often partially obscured"): -10,
        _norm_value("Face mostly not visible"): -20,
    }
    if face_visibility_norm in face_visibility_deltas:
        record(
            "Visual Analysis",
            "Face Visibility Quality",
            face_visibility,
            face_visibility_deltas[face_visibility_norm],
        )

    photo_editing = visual_val("Photo Authenticity / Editing Level")
    photo_editing_norm = _norm_value(photo_editing)
    photo_editing_deltas = {
        _norm_value("No obvious filters"): 0,
        _norm_value("Some filters or mild editing"): -5,
        _norm_value("Heavy filters/face smoothing"): -20,
        _norm_value("Unclear"): 0,
    }
    if photo_editing_norm in photo_editing_deltas:
        record(
            "Visual Analysis",
            "Photo Authenticity / Editing Level",
            photo_editing,
            photo_editing_deltas[photo_editing_norm],
        )

    body_fat = visual_val("Apparent Body Fat Level")
    body_fat_norm = _norm_value(body_fat)
    body_fat_deltas = {
        _norm_value("Low"): 0,
        _norm_value("Average"): 0,
        _norm_value("High"): -10,
        _norm_value("Very high"): -1000,
        _norm_value("Unclear"): 0,
    }
    if body_fat_norm in body_fat_deltas:
        record(
            "Visual Analysis",
            "Apparent Body Fat Level",
            body_fat,
            body_fat_deltas[body_fat_norm],
        )

    distinctiveness = visual_val("Profile Distinctiveness")
    distinctiveness_norm = _norm_value(distinctiveness)
    distinctiveness_deltas = {
        _norm_value("High (specific/unique)"): 5,
        _norm_value("Medium"): 0,
        _norm_value("Low (generic/boilerplate)"): -5,
        _norm_value("Unclear"): 0,
    }
    if distinctiveness_norm in distinctiveness_deltas:
        record(
            "Visual Analysis",
            "Profile Distinctiveness",
            distinctiveness,
            distinctiveness_deltas[distinctiveness_norm],
        )

    attire = visual_val("Attire and Style Indicators")
    attire_norm = _norm_value(attire)
    if attire_norm in {
        _norm_value("Form-fitting/suggestive"),
        _norm_value("Highly revealing"),
        _norm_value("Edgy/alternative"),
    }:
        record("Visual Analysis", "Attire and Style Indicators", attire, +20)

    attractiveness = visual_val("Apparent Attractiveness Tier")
    attractiveness_norm = _norm_value(attractiveness)
    attractiveness_deltas = {
        _norm_value("Very unattractive/morbidly obese"): -1000,
        _norm_value("Low"): -1000,
        _norm_value("Average"): -20,
        _norm_value("Above average"): 5,
        _norm_value("High"): 10,
        _norm_value("Very attractive"): 20,
        _norm_value("Extremely attractive"): 30,
        _norm_value("Supermodel"): 40,
    }
    if attractiveness_norm in attractiveness_deltas:
        att_delta = attractiveness_deltas[attractiveness_norm]
        if face_visibility_norm != _norm_value("Clear face in 3+ photos"):
            att_delta = min(att_delta, 5)
        if photo_editing_norm == _norm_value("Heavy filters/face smoothing"):
            att_delta = min(att_delta, 0)
        record(
            "Visual Analysis",
            "Apparent Attractiveness Tier",
            attractiveness,
            att_delta,
        )

    symmetry = visual_val("Facial Symmetry Level")
    symmetry_norm = _norm_value(symmetry)
    if symmetry_norm == _norm_value("Low"):
        record("Visual Analysis", "Facial Symmetry Level", symmetry, -1000)
    elif symmetry_norm == _norm_value("Moderate"):
        record("Visual Analysis", "Facial Symmetry Level", symmetry, -20)

    hair_color = visual_val("Hair Color")
    hair_norm = _norm_value(hair_color)
    if hair_norm == _norm_value("Red/ginger"):
        record("Visual Analysis", "Hair Color", hair_color, +20)
    elif hair_norm in {
        _norm_value("Dyed blue"),
        _norm_value("Dyed pink"),
        _norm_value("Dyed (unnatural other)"),
        _norm_value("Dyed (mixed/multiple colors)"),
    }:
        record("Visual Analysis", "Hair Color", hair_color, +10)

    piercing = visual_val("Visible Piercing Level")
    piercing_norm = _norm_value(piercing)
    if piercing_norm == _norm_value("High"):
        record("Visual Analysis", "Visible Piercing Level", piercing, -1000)
    elif piercing_norm == _norm_value("Moderate"):
        record("Visual Analysis", "Visible Piercing Level", piercing, -20)
    elif piercing_norm == _norm_value("None visible"):
        record("Visual Analysis", "Visible Piercing Level", piercing, +5)

    build = visual_val("Apparent Build Category")
    build_norm = _norm_value(build)
    if build_norm == _norm_value("Obese/high body fat"):
        record("Visual Analysis", "Apparent Build Category", build, -1000)
    elif build_norm == _norm_value("Curvy (softer proportions)"):
        record("Visual Analysis", "Apparent Build Category", build, -10)
    elif build_norm == _norm_value("Muscular/built"):
        record("Visual Analysis", "Apparent Build Category", build, +10)

    skin = visual_val("Apparent Skin Tone")
    skin_norm = _norm_value(skin)
    if skin_norm in {_norm_value("Golden/medium-brown"), _norm_value("Warm brown/deep tan")}:
        record("Visual Analysis", "Apparent Skin Tone", skin, -20)
    elif skin_norm in {_norm_value("Dark-brown/chestnut"), _norm_value("Very dark/ebony/deep")}:
        record("Visual Analysis", "Apparent Skin Tone", skin, -1000)

    upper_body = visual_val("Apparent Upper Body Proportions")
    upper_norm = _norm_value(upper_body)
    if upper_norm == _norm_value("Petite/small/narrow"):
        record("Visual Analysis", "Apparent Upper Body Proportions", upper_body, -5)
    elif upper_norm and upper_norm != _norm_value("Average/balanced/proportional"):
        record("Visual Analysis", "Apparent Upper Body Proportions", upper_body, +5)

    enhancements = _split_csv(visual_val("Visible Enhancements or Features"))
    for item in enhancements:
        item_norm = _norm_value(item)
        if item_norm == _norm_value("Glasses"):
            record("Visual Analysis", "Visible Enhancements or Features", item, +5)
        elif item_norm == _norm_value("Makeup (heavy)"):
            record("Visual Analysis", "Visible Enhancements or Features", item, -10)
        elif item_norm == _norm_value("Very long nails (2cm+)"):
            record("Visual Analysis", "Visible Enhancements or Features", item, -10)
        elif item_norm == _norm_value("False eyelashes (obvious)"):
            record("Visual Analysis", "Visible Enhancements or Features", item, -5)

    red_flags = _split_csv(visual_val("Presentation Red Flags"))
    for flag in red_flags:
        flag_norm = _norm_value(flag)
        if flag_norm == _norm_value("None"):
            continue
        if flag_norm == _norm_value("Heavy filters/face smoothing") and photo_editing_norm == _norm_value("Heavy filters/face smoothing"):
            continue
        record("Visual Analysis", "Presentation Red Flags", flag, -5)

    # Profile Evaluation (LLM2)
    job_band = _norm_value((eval_result.get("job") or {}).get("band", ""))
    if job_band == _norm_value("T0"):
        record("Profile Eval", "Job Tier", "T0", -10)
    elif job_band == _norm_value("T1"):
        record("Profile Eval", "Job Tier", "T1", -5)
    elif job_band == _norm_value("T3"):
        record("Profile Eval", "Job Tier", "T3", +5)
    elif job_band == _norm_value("T4"):
        record("Profile Eval", "Job Tier", "T4", +10)

    university_elite = int(eval_result.get("university_elite", 0) or 0)
    if university_elite == 1:
        record("Profile Eval", "University Elite", "Yes", +10)

    # Age delta (logged only)
    apparent_age = visual_val("Apparent Age Range Category")
    apparent_mid = AGE_RANGE_MIDPOINTS.get(apparent_age) if isinstance(apparent_age, str) else None
    apparent_bounds = AGE_RANGE_BOUNDS.get(apparent_age) if isinstance(apparent_age, str) else None
    age_delta = None
    if declared_age_int is not None and apparent_mid is not None:
        in_range = False
        if apparent_bounds is not None:
            low, high = apparent_bounds
            if high is None:
                in_range = declared_age_int >= low
            else:
                in_range = low <= declared_age_int <= high
        age_delta = 0.0 if in_range else round(float(apparent_mid) - float(declared_age_int), 2)

    score_total = sum(c["delta"] for c in contribs)

    home_iso = str(eval_result.get("home_country_iso", "") or "").upper().strip()
    return {
        "score": int(score_total),
        "hard_kills": hard_kills,
        "contributions": contribs,
        "signals": {
            "declared_age": declared_age_int,
            "apparent_age_range": apparent_age if isinstance(apparent_age, str) else "",
            "apparent_age_range_midpoint": apparent_mid,
            "age_delta": age_delta,
        },
        "profile_eval_inputs": {
            "job_band": (eval_result.get("job") or {}).get("band", ""),
            "university_elite": university_elite,
            "home_country_iso": home_iso,
        },
    }


def _format_score_table(label: str, score_result: Dict[str, Any]) -> str:
    contribs = score_result.get("contributions", []) if isinstance(score_result, dict) else []
    signals = (score_result.get("signals", {}) if isinstance(score_result, dict) else {}) or {}
    profile_eval_inputs = (score_result.get("profile_eval_inputs", {}) if isinstance(score_result, dict) else {}) or {}
    hard_kills = score_result.get("hard_kills", []) if isinstance(score_result, dict) else []
    score = score_result.get("score", 0) if isinstance(score_result, dict) else 0

    lines: List[str] = []
    lines.append(f"{label.upper()} SCORE SUMMARY")
    lines.append(f"Final score: {score}")
    lines.append(f"Hard kills: {len(hard_kills)}")
    lines.append(
        "Age signals: declared_age={declared_age} apparent_midpoint={apparent_age_range_midpoint} age_delta={age_delta}".format(
            declared_age=signals.get("declared_age"),
            apparent_age_range_midpoint=signals.get("apparent_age_range_midpoint"),
            age_delta=signals.get("age_delta"),
        )
    )
    lines.append(
        "Eval signals: job_band={job_band} university_elite={university_elite} home_country_iso={home_country_iso}".format(
            job_band=profile_eval_inputs.get("job_band"),
            university_elite=profile_eval_inputs.get("university_elite"),
            home_country_iso=profile_eval_inputs.get("home_country_iso"),
        )
    )
    lines.append("")
    lines.append("CONTRIBUTIONS (non-zero)")
    if not contribs:
        lines.append("(none)")
        return "\n".join(lines)

    headers = ["Section", "Field", "Value", "Delta"]
    rows = [[c.get("section", ""), c.get("field", ""), str(c.get("value", "")), str(c.get("delta", ""))] for c in contribs]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, col in enumerate(row):
            widths[i] = max(widths[i], len(col))

    def fmt_row(row: List[str]) -> str:
        return " | ".join(col.ljust(widths[i]) for i, col in enumerate(row))

    lines.append(fmt_row(headers))
    lines.append("-+-".join("-" * w for w in widths))
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


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
    prompt_engine.build_structured_profile_prompt = LLM1

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
    eval_result = run_profile_eval_llm(
        extracted,
        model=os.getenv("LLM_SMALL_MODEL") or os.getenv("GEMINI_SMALL_MODEL") or None,
    )
    long_score_result = _score_profile_long(extracted, eval_result)
    short_score_result = _score_profile_short(extracted, eval_result)
    score_table_long = _format_score_table("Long", long_score_result)
    score_table_short = _format_score_table("Short", short_score_result)
    score_table = score_table_long + "\n\n" + score_table_short
    out = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "llm_provider": os.getenv("LLM_PROVIDER", ""),
            "model": os.getenv("LLM_SMALL_MODEL") or os.getenv("GEMINI_SMALL_MODEL") or "",
            "images_count": llm_meta.get("images_count"),
            "images_paths": llm_meta.get("images_paths", []),
            "timings": s.get("timings", {}),
            "scoring_ruleset": "long_v0",
        },
        "extracted_profile": extracted,
        "profile_eval": eval_result,
        "long_score_result": long_score_result,
        "short_score_result": short_score_result,
        "score_table_long": score_table_long,
        "score_table_short": score_table_short,
        "score_table": score_table,
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n" + score_table)

    out_path = ""
    try:
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("logs", f"rating_test_{ts}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        table_path = os.path.join("logs", f"rating_test_{ts}.txt")
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(score_table)
        print(f"Wrote results to {out_path}")
        print(f"Wrote score table to {table_path}")
    except Exception as e:
        print(f"Failed to write results: {e}")

    # TEMP: manual preference logging (remove after tuning)
    try:
        name = (
            (extracted.get("Core Biometrics (Objective)", {}) or {}).get("Name")
            if isinstance(extracted, dict)
            else ""
        )
        long_score = long_score_result.get("score", 0) if isinstance(long_score_result, dict) else 0
        short_score = short_score_result.get("score", 0) if isinstance(short_score_result, dict) else 0
        print("\n=== Manual Preference Input (TEMP) ===")
        def _prompt_yes_no(label: str) -> str:
            while True:
                resp = input(label).strip().lower()
                if resp in {"y", "n"}:
                    return resp
                print("Please enter y or n.")

        def _prompt_short_long(label: str) -> str:
            while True:
                resp = input(label).strip().lower()
                if resp in {"short"}:
                    return "short"
                if resp in {"long"}:
                    return "long"
                print("Please enter exactly 'short' or 'long'.")

        user_like = _prompt_yes_no("Your verdict? (y/n): ")
        user_score_raw = input("Your score (0-10): ").strip()
        user_score = None
        try:
            user_score = int(user_score_raw)
        except Exception:
            user_score = None
        short_long = "n/a"
        if user_like == "y":
            short_long = _prompt_short_long("Short or long? (type exactly 'short' or 'long'): ")
        thoughts = input("Any thoughts? (optional): ").strip()

        long_contributions = long_score_result.get("contributions", []) if isinstance(long_score_result, dict) else []
        short_contributions = short_score_result.get("contributions", []) if isinstance(short_score_result, dict) else []
        age = None
        if isinstance(long_score_result, dict):
            age = (long_score_result.get("signals", {}) or {}).get("declared_age")
        if age is None and isinstance(extracted, dict):
            age = (extracted.get("Core Biometrics (Objective)", {}) or {}).get("Age")

        manual_log = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "name": name or "",
            "json_path": out_path or "",
            "long_score": int(long_score),
            "short_score": int(short_score),
            "user_like": user_like,
            "user_score": user_score,
            "short_long": short_long,
            "age": age,
            "long_score_contributions": long_contributions,
            "short_score_contributions": short_contributions,
            "thoughts": thoughts,
        }
        manual_log_path = os.path.join("logs", "manual_preference_log.jsonl")
        with open(manual_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(manual_log, ensure_ascii=False) + "\n")
        print(f"Logged manual preferences to {manual_log_path}")
    except Exception as e:
        print(f"Failed to log manual preferences: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
