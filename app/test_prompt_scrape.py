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
from helper_functions import (
    ensure_adb_running,
    capture_screenshot,
    swipe,
    tap,
    detect_like_button_cv,
    are_images_similar,
    are_images_similar_roi,
)
from hinge_agent import HingeAgent, HingeAgentState
from llm_client import get_default_model, get_llm_client, resolve_model
from text_utils import normalize_dashes


def LLM1(image_paths: List[str] | None = None) -> str:
    image_list_block = ""
    if image_paths:
        basenames = [os.path.basename(p) for p in image_paths if isinstance(p, str)]
        if basenames:
            image_list_block = (
                "\nAvailable vertical page screenshots (use these for source_file):\n- "
                + "\n- ".join(basenames)
                + "\n\n"
            )
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
        '      {"id": "prompt_1", "prompt": "", "answer": "", "source_file": "", "page_half": ""},\n'
        '      {"id": "prompt_2", "prompt": "", "answer": "", "source_file": "", "page_half": ""},\n'
        '      {"id": "prompt_3", "prompt": "", "answer": "", "source_file": "", "page_half": ""}\n'
        '    ],\n'
        '    "Poll (optional, most profiles will not have this)": {\n'
        '      "id": "poll_1",\n'
        '      "question": "",\n'
        '      "source_file": "",\n'
        '      "page_half": "",\n'
        '      "answers": [\n'
        '        {"id": "poll_1_a", "text": ""},\n'
        '        {"id": "poll_1_b", "text": ""},\n'
        '        {"id": "poll_1_c", "text": ""}\n'
        '      ]\n'
        '    },\n'
        '    "Other text on profile not covered by above": "",\n'
        '    "Description of any non-photo media (e.g., video (identified via timestamp in top right), voice note)": "",\n'
        '    "Extensive Description of Photo 1": {"id": "photo_1", "description": "", "source_file": "", "page_half": ""},\n'
        '    "Extensive Description of Photo 2": {"id": "photo_2", "description": "", "source_file": "", "page_half": ""},\n'
        '    "Extensive Description of Photo 3": {"id": "photo_3", "description": "", "source_file": "", "page_half": ""},\n'
        '    "Extensive Description of Photo 4": {"id": "photo_4", "description": "", "source_file": "", "page_half": ""},\n'
        '    "Extensive Description of Photo 5": {"id": "photo_5", "description": "", "source_file": "", "page_half": ""},\n'
        '    "Extensive Description of Photo 6": {"id": "photo_6", "description": "", "source_file": "", "page_half": ""}\n'
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
        '      "Visible Piercing Level": "",\n'
        '      "Short-Term / Hookup Orientation Signals": ""\n'
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
        "2b) Target IDs (for downstream LLMs):\n"
        "- Include IDs directly inside prompts, poll, and photo description objects.\n"
        "- IDs must be exactly: prompt_1..prompt_3, photo_1..photo_6, poll_1, poll_1_a, poll_1_b, poll_1_c.\n"
        "- For prompts, polls, and photos, include source_file (must be one of the vertical page screenshot filenames provided) and page_half (must be exactly \"top\" or \"bottom\").\n"
        "- If no poll is present, still include poll_1 with empty question/answers.\n\n"
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
        '- "Dating Intentions": "Life partner", "Long-term relationship", "Long-term relationship, open to short", "Short-term relationship, open to long", "Short-term relationship", "Figuring out my dating goals".\n'
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
        '"Apparent Attractiveness Tier (CRITICAL - CONSERVATIVE SCORING REQUIRED)": Default to "Average". This is a mainstream/common-judgment rating. If uncertain between two tiers, ALWAYS choose the LOWER tier. If "Face Visibility Quality" is NOT "Clear face in 3+ photos", do NOT output tiers ABOVE "Above average". For tiers ABOVE "Average", require clear evidence across multiple photos; do not promote based on vibes, style, or distinctiveness alone. If multiple clear limiting factors are present, do not avoid using "Low" or "Very unattractive/morbidly obese". Use "Supermodel" only in extremely rare cases where the subject clearly stands out far above typical dating-app profiles across multiple photos (exceptional facial symmetry, consistently flattering angles/lighting, and polished grooming). This rating must reflect mainstream Western heterosexual dating-app standards for women. To be rated ABOVE "Average", the subject should broadly fit common feminine beauty norms in facial structure, body proportions, and overall presentation; profiles that significantly diverge from these norms should not be rated above "Average" regardless of styling, grooming, or effort.\n\n'
        '"Reasoning for attractiveness tier": Provide ONE concise sentence listing observable factors. Requirements by tier: if tier is ABOVE "Average", include at least 1 clear positive AND 1 clear limiting factor; if tier is "Average", include at least 1 positive AND 1 limiting factor; if tier is "Low" or "Very unattractive/morbidly obese", include 2 or more limiting factors (positives optional). Avoid vague words.\n\n'
        '"Facial Symmetry Level (CONSERVATIVE)": Default to "Moderate". Use "High" or "Very high" only when symmetry is clearly and consistently strong across multiple photos. Do NOT infer symmetry from grooming or styling.\n\n'
        '"Apparent Age Range Category (IMPORTANT - VISUAL ONLY)": This field must be based ONLY on facial appearance in the photos. Do NOT reconcile or align with the stated age. Disagreement is expected and desirable. If uncertain between adjacent age ranges, ALWAYS choose the OLDER range. Overestimating apparent age is preferred to underestimating it.\n\n'
        '"Facial Proportion Balance": One of: "Balanced/proportional", "Slightly unbalanced", "Noticeably unbalanced"\n\n'
        '"Grooming Effort Level": One of: "Minimal/natural", "Moderate/casual", "High/polished", "Heavy/overdone"\n\n'
        '"Presentation Red Flags": Pick any that apply (comma-separated): "None", "Poor lighting", "Blurry/low resolution", "Unflattering angle", "Heavy filters/face smoothing", "Sunglasses in most photos", "Face mostly obscured", "Group photos unclear who is the subject", "Too many distant shots", "Mirror selfie cluttered", "Messy background", "Only one clear solo photo", "Awkward cropping", "Overexposed/washed out", "Inconsistent appearance across photos", "Entitlement language in prompts", "Transactional dating expectations stated explicitly", "Rigid gender role expectations stated explicitly", "Aggressive negativity in prompts", "Excessive rules or dealbreakers listed", "Hostile or contemptuous humor in prompts", "Passive-aggressive tone in prompts", "Explicit materialism or status demands"\n\n'
        '"Red Flags Notes: This must appear in most or all pictures to be red flagged. One or two group photos or filtered photos out of 6 is acceptable. 4 or 5 is not.'
        '"Visible Tattoo Level": One of: "None visible", "Small/minimal", "Moderate", "High"\n\n'
        '"Visible Piercing Level": One of: "None visible", "Minimal", "Moderate", "High"\n'
        '"Rule (piercings): "Minimal" = ONLY standard single earlobe earrings. ANY septum/nose/eyebrow/lip or multiple ear piercings MUST be "Moderate" or "High". If uncertain between adjacent levels, choose the HIGHER level."\n\n'
        '"Short-Term / Hookup Orientation Signals": One of: "None evident", "Low", "Moderate", "High"\n\n'
        '"Short-Term / Hookup Orientation Signalling": Assess whether the profile signals casual sexual openness based on a combination of visual AND textual cues, including repeated suggestive attire or poses, sexually loaded prompt answers or innuendo, emphasis on physical touch, flirtatious framing, or body-forward presentation. Subtle but consistent signals across the profile should be rated "Moderate" rather than "Low".'
    ) + image_list_block






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


def LLM3_LONG(extracted: Dict[str, Any]) -> str:
    extracted_json = json.dumps(extracted or {}, ensure_ascii=False, indent=2)
    return (
        "You are generating opening messages for Hinge.\n"
        "These should be warm, natural, and genuinely curious questions — the kind of thing a normal attractive guy would send.\n"
        "The goal is to start an easy conversation.\n\n"
        "Profile details:\n"
        f"{extracted_json}\n\n"
        "Task: Generate exactly 10 DISTINCT opening messages as JSON.\n\n"
        "Rules:\n"
        "- Output JSON only, no extra text.\n"
        "- Each opener must anchor to EXACTLY ONE profile element ID: prompt_1..prompt_3, photo_1..photo_6, poll_1_a|poll_1_b|poll_1_c.\n"
        "- Every opener must be a simple question (or light A/B choice) that is easy to reply to.\n"
        "- Keep it sweet, and either curious or flirty\n"
        "- Do NOT narrate the photo like a caption. Reference it naturally (\"where was this?\" not \"in photo 3\"). The text will be linked to the photo by the user, using the index supplied.\n"
        "- Avoid generic compliments like \"you’re stunning\".\n"
        "- Avoid forced jokes, try-hard banter, or anything that sounds scripted.\n"
        "- Messages should feel human, relaxed, and effortless. Don't overly specify - i.e. \"you look amazing in that red fabric dress\" vs \"you look amazing in that dress\". \n\n"
        "Output format (JSON only):\n"
        "{\n"
        '  "openers": [\n'
        "    {\n"
        '      "text": \"...\",\n'
        '      "main_target_type": \"prompt|photo|poll\",\n'
        '      "main_target_id": \"prompt_1\",\n'
        '      "hook_basis": \"short internal note on what you targeted\"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )



def LLM3_SHORT(extracted: Dict[str, Any]) -> str:
    extracted_json = json.dumps(extracted or {}, ensure_ascii=False, indent=2)
    return (
        "You are generating opening messages for Hinge.\n"
        "These should be confident, flirty, and optionally sexually charged when the profile vibe supports it.\n"
        "The goal is to create immediate chemistry and get a reply.\n\n"
        "Profile details:\n"
        f"{extracted_json}\n\n"
        "Task: Generate exactly 10 DISTINCT opening messages as JSON.\n\n"
        "Rules:\n"
        "- Output JSON only, no extra text.\n"
        "- Each opener must anchor to EXACTLY ONE profile element ID: prompt_1..prompt_3, photo_1..photo_6, poll_1_a|poll_1_b|poll_1_c.\n"
        "- Every opener must be a question (or light A/B choice) that is easy to reply to.\n"
        "- Keep it bold, playful, and flirty. If the profile signals a spicy vibe, add sexual tension or innuendo.\n"
        "- Do NOT narrate the photo like a caption. Reference it naturally (\"where was this?\" not \"in photo 3\"). The text will be linked to the photo by the user, using the index supplied.\n"
        "- Avoid generic compliments like \"you’re stunning\".\n"
        "- Avoid try-hard banter, forced jokes, or anything that sounds scripted.\n"
        "- Keep it human and natural, not verbose.\n"
        "Output format (JSON only):\n"
        "{\n"
        '  "openers": [\n'
        "    {\n"
        '      "text": \"...\",\n'
        '      "main_target_type": \"prompt|photo|poll\",\n'
        '      "main_target_id": \"prompt_1\",\n'
        '      "hook_basis": \"short internal note on what you targeted\"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )

def LLM4(openers_json: Dict[str, Any]) -> str:
    openers_str = json.dumps(openers_json or {}, ensure_ascii=False, indent=2)
    return (
        "You are selecting the single best Hinge opener from a provided list.\n"
        "Pick the one most likely to get a reply. Be decisive.\n\n"
        "Openers JSON:\n"
        f"{openers_str}\n\n"
        "Output JSON only:\n"
        "{\n"
        '  "chosen_index": 0,\n'
        '  "chosen_text": "",\n'
        '  "main_target_type": "prompt|photo|poll",\n'
        '  "main_target_id": "",\n'
        '  "rationale": ""\n'
        "}\n"
    )



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


def run_llm3_long(extracted: Dict[str, Any], model: str | None = None) -> Dict[str, Any]:
    prompt = LLM3_LONG(extracted)
    requested_model = model or get_default_model()
    resolved_model = resolve_model(requested_model)
    try:
        resp = get_llm_client().chat.completions.create(
            model=resolved_model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def run_llm3_short(extracted: Dict[str, Any], model: str | None = None) -> Dict[str, Any]:
    prompt = LLM3_SHORT(extracted)
    requested_model = model or get_default_model()
    resolved_model = resolve_model(requested_model)
    try:
        resp = get_llm_client().chat.completions.create(
            model=resolved_model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def run_llm4(openers_json: Dict[str, Any], model: str | None = None) -> Dict[str, Any]:
    prompt = LLM4(openers_json)
    requested_model = model or get_default_model()
    resolved_model = resolve_model(requested_model)
    try:
        resp = get_llm_client().chat.completions.create(
            model=resolved_model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _resolve_source_file(source_file: str, images_paths: List[str]) -> str:
    if not source_file:
        return ""
    src = str(source_file).strip()
    if not src:
        return ""
    # If absolute path exists, keep it.
    if os.path.exists(src):
        return os.path.abspath(src)
    # Try basename match against known images paths.
    base = os.path.basename(src)
    for p in images_paths or []:
        if os.path.basename(str(p)) == base:
            return os.path.abspath(p)
    # Heuristic: "page_3_top.png" -> vpage_3
    import re
    m = re.search(r"page[_-]?(\d+)", base, re.IGNORECASE)
    if m:
        idx = m.group(1)
        for p in images_paths or []:
            if f"_vpage_{idx}.png" in str(p):
                return os.path.abspath(p)
    return ""


def _normalize_text_basic(text: str) -> str:
    import re
    s = (text or "").lower()
    s = normalize_dashes(s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())


def _target_log(message: str) -> None:
    if os.getenv("HINGE_TARGET_DEBUG", "1") == "1":
        print(message)


def _ocr_find_text_source_file(images_paths: List[str], target_text: str) -> Dict[str, Any]:
    try:
        import cv2
        import pytesseract
    except Exception:
        _target_log("[OCR] pytesseract missing; OCR disabled")
        return {
            "source_file": "",
            "page_half": "unknown",
            "confidence": 0.0,
            "error": "pytesseract_missing",
        }

    target_norm = _normalize_text_basic(target_text)
    if not target_norm:
        return {"source_file": "", "page_half": "unknown", "confidence": 0.0}
    target_words = {w for w in target_norm.split() if len(w) > 2}
    if not target_words:
        return {"source_file": "", "page_half": "unknown", "confidence": 0.0}

    _target_log(
        f"[OCR] start target='{target_text.strip()[:80]}' words={len(target_words)}"
    )
    for path in images_paths:
        if "vpage_" not in str(path):
            continue
        abs_path = os.path.abspath(path)
        try:
            img = cv2.imread(abs_path)
            if img is None:
                continue
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            words = data.get("text", []) or []
            xs = data.get("left", []) or []
            ys = data.get("top", []) or []
            hs = data.get("height", []) or []
            h_img = img.shape[0]
            matched_ys: List[int] = []
            matched = 0
            for w, y, h in zip(words, ys, hs):
                w_norm = _normalize_text_basic(w)
                if w_norm and w_norm in target_words:
                    matched += 1
                    matched_ys.append(int(y + (h / 2)))
            _target_log(
                f"[OCR] scan {os.path.basename(abs_path)} matched={matched}/{len(target_words)}"
            )
            if matched >= 2:
                avg_y = sum(matched_ys) / max(len(matched_ys), 1)
                page_half = "top" if avg_y < (h_img / 2) else "bottom"
                confidence = min(1.0, matched / max(len(target_words), 1))
                _target_log(
                    f"[OCR] match {os.path.basename(abs_path)} half={page_half} conf={confidence:.2f}"
                )
                return {
                    "source_file": abs_path,
                    "page_half": page_half,
                    "confidence": confidence,
                }
        except Exception:
            continue
    _target_log("[OCR] no match")
    return {"source_file": "", "page_half": "unknown", "confidence": 0.0}


def _resolve_target_from_extracted(
    extracted: Dict[str, Any],
    target_id: str,
    images_paths: List[str],
) -> Dict[str, Any]:
    target_id = (target_id or "").strip()
    if target_id.startswith("photo_"):
        content = (extracted or {}).get("Profile Content (Free Description)", {}) or {}
        for i in range(1, 7):
            key = f"Extensive Description of Photo {i}"
            entry = content.get(key, {}) if isinstance(content, dict) else {}
            if isinstance(entry, dict) and entry.get("id") == target_id:
                src = _resolve_source_file(entry.get("source_file", ""), images_paths)
                page_half = entry.get("page_half", "")
                return {
                    "type": "photo",
                    "id": target_id,
                    "source_file": src,
                    "page_half": page_half or "",
                }
        return {"type": "photo", "id": target_id, "source_file": "", "page_half": ""}

    if target_id.startswith("prompt_"):
        content = (extracted or {}).get("Profile Content (Free Description)", {}) or {}
        prompts = content.get("Profile Prompts and Answers", []) if isinstance(content, dict) else []
        for item in prompts or []:
            if isinstance(item, dict) and item.get("id") == target_id:
                text = str(item.get("prompt") or "").strip()
                src = _resolve_source_file(item.get("source_file", ""), images_paths)
                page_half = item.get("page_half", "")
                if text:
                    res = _ocr_find_text_source_file(images_paths, text)
                    if res.get("source_file"):
                        src = res.get("source_file", "")
                        page_half = res.get("page_half", "")
                return {
                    "type": "prompt",
                    "id": target_id,
                    "prompt_text": text,
                    "source_file": src,
                    "page_half": page_half,
                }
        return {"type": "prompt", "id": target_id, "prompt_text": "", "source_file": "", "page_half": ""}

    if target_id.startswith("poll_1_"):
        content = (extracted or {}).get("Profile Content (Free Description)", {}) or {}
        poll = content.get("Poll (optional, most profiles will not have this)", {}) if isinstance(content, dict) else {}
        answers = poll.get("answers", []) if isinstance(poll, dict) else []
        for ans in answers or []:
            if isinstance(ans, dict) and ans.get("id") == target_id:
                text = str(ans.get("text") or "").strip()
                src = _resolve_source_file(poll.get("source_file", ""), images_paths)
                page_half = poll.get("page_half", "")
                if text:
                    res = _ocr_find_text_source_file(images_paths, text)
                    if res.get("source_file"):
                        src = res.get("source_file", "")
                        page_half = res.get("page_half", "")
                return {
                    "type": "poll",
                    "id": target_id,
                    "poll_response": text,
                    "source_file": src,
                    "page_half": page_half,
                }
        return {"type": "poll", "id": target_id, "poll_response": "", "source_file": "", "page_half": ""}

    return {"type": "unknown", "id": target_id}


def _reset_to_top(device, width: int, height: int, attempts: int = 3) -> None:
    for _ in range(attempts):
        swipe(device, int(width * 0.5), int(height * 0.3), int(width * 0.5), int(height * 0.85), 500)
        time.sleep(0.5)


def _seek_to_source_file(
    device,
    width: int,
    height: int,
    source_file: str,
    page_half: str,
    max_swipes: int = 12,
    direction: str = "down",
) -> str:
    if not source_file:
        return ""
    source_file = os.path.abspath(source_file)
    page_half = (page_half or "").strip().lower()
    y_center = int(height * (0.25 if page_half == "top" else 0.75))
    _target_log(
        f"[HASH] seek start source={os.path.basename(source_file)} page_half={page_half or 'unknown'} direction={direction}"
    )
    for _ in range(max_swipes):
        ts = int(time.time() * 1000)
        cur_path = capture_screenshot(device, f"seek_{ts}")
        matched = False
        if page_half in {"top", "bottom"}:
            matched = are_images_similar_roi(cur_path, source_file, y_center=y_center, threshold=6)
        else:
            matched = are_images_similar(cur_path, source_file, threshold=6)
        _target_log(
            f"[HASH] attempt {os.path.basename(cur_path)} match={'yes' if matched else 'no'}"
        )
        if matched:
            return cur_path
        if direction == "up":
            swipe(device, int(width * 0.5), int(height * 0.2), int(width * 0.5), int(height * 0.8), 500)
        else:
            swipe(device, int(width * 0.5), int(height * 0.8), int(width * 0.5), int(height * 0.2), 500)
        time.sleep(0.6)
    return ""


def _click_like_for_target(device, width: int, height: int, page_half: str = "") -> bool:
    ts = int(time.time() * 1000)
    cur_path = capture_screenshot(device, f"like_seek_{ts}")
    cv_res = detect_like_button_cv(cur_path)
    if isinstance(cv_res, dict) and cv_res.get("found"):
        like_x = int(cv_res["x"])
        like_y = int(cv_res["y"])
        tap(device, like_x, like_y)
        return True

    # Fallback: tap right side of the relevant half
    page_half = (page_half or "").strip().lower()
    if page_half == "top":
        tap(device, int(width * 0.92), int(height * 0.35))
        return True
    if page_half == "bottom":
        tap(device, int(width * 0.92), int(height * 0.75))
        return True
    return False


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


# Long Weightings
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

    short_term = visual_val("Short-Term / Hookup Orientation Signals")
    short_term_norm = _norm_value(short_term)
    if short_term_norm == _norm_value("High"):
        record("Visual Analysis", "Short-Term / Hookup Orientation Signals", short_term, -5)

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


# Short Weightings
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
    dating_deltas = {
        _norm_value("Life partner"): -5,
        _norm_value("Long-term relationship, open to short"): +10,
        _norm_value("Short-term relationship"): +15,
        _norm_value("Short-term relationship, open to long"): +10,
        _norm_value("Figuring out my dating goals"): +10,
    }
    if dating_norm in dating_deltas:
        record("Core Biometrics", "Dating Intentions", dating, dating_deltas[dating_norm])

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

    # Age weighting (declared age only)
    declared_age = core_val("Age")
    declared_age_int = None
    try:
        declared_age_int = int(declared_age) if declared_age is not None and str(declared_age).strip() != "" else None
    except Exception:
        declared_age_int = None
    if declared_age_int is not None:
        if 18 <= declared_age_int <= 22:
            record("Core Biometrics", "Age", "18-22", +5)
        elif 36 <= declared_age_int <= 40:
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

    overall_vibe = visual_val("Overall Visual Appeal Vibe")
    overall_vibe_norm = _norm_value(overall_vibe)
    if overall_vibe_norm in {
        _norm_value("Playful/flirty"),
        _norm_value("Sensual/alluring"),
    }:
        record("Visual Analysis", "Overall Visual Appeal Vibe", overall_vibe, +5)
    elif overall_vibe_norm == _norm_value("Very low-key/understated"):
        record("Visual Analysis", "Overall Visual Appeal Vibe", overall_vibe, -5)

    attire = visual_val("Attire and Style Indicators")
    attire_norm = _norm_value(attire)
    if attire_norm == _norm_value("Very modest/covered"):
        record("Visual Analysis", "Attire and Style Indicators", attire, -10)
    elif attire_norm in {
        _norm_value("Form-fitting/suggestive"),
        _norm_value("Highly revealing"),
        _norm_value("Edgy/alternative"),
    }:
        record("Visual Analysis", "Attire and Style Indicators", attire, +10)

    body_language = visual_val("Body Language and Expression")
    body_language_norm = _norm_value(body_language)
    if body_language_norm in {
        _norm_value("Confident/engaging"),
        _norm_value("Playful/flirty"),
    }:
        record("Visual Analysis", "Body Language and Expression", body_language, +5)

    fitness = _split_csv(visual_val("Indicators of Fitness or Lifestyle"))
    for item in fitness:
        item_norm = _norm_value(item)
        if item_norm in {
            _norm_value("Visible muscle tone"),
            _norm_value("Athletic poses"),
        }:
            record("Visual Analysis", "Indicators of Fitness or Lifestyle", item, +10)

    short_term = visual_val("Short-Term / Hookup Orientation Signals")
    short_term_norm = _norm_value(short_term)
    short_term_deltas = {
        _norm_value("Low"): 5,
        _norm_value("Moderate"): 10,
        _norm_value("High"): 15,
    }
    if short_term_norm in short_term_deltas:
        record(
            "Visual Analysis",
            "Short-Term / Hookup Orientation Signals",
            short_term,
            short_term_deltas[short_term_norm],
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

    grooming = visual_val("Grooming Effort Level")
    if _norm_value(grooming) == _norm_value("Minimal/natural"):
        record("Visual Analysis", "Grooming Effort Level", grooming, -5)

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
    # Quiet/compact: disable verbose CV + console debug unless explicitly overridden.
    os.environ.setdefault("HINGE_VERBOSE_LOGGING", "0")
    os.environ.setdefault("HINGE_CV_DEBUG_MODE", "0")
    os.environ.setdefault("HINGE_TARGET_DEBUG", "1")

    # Suppress extraction warnings in console unless explicitly enabled.
    os.environ.setdefault("HINGE_SHOW_EXTRACTION_WARNINGS", "0")


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

    # Inject vertical image list into LLM1 prompt at batch-build time.
    try:
        original_build_llm_batch_payload = ha.build_llm_batch_payload

        def _build_llm_batch_payload_with_images(screenshots: List[str], prompt: str | None = None):
            vpages = [p for p in (screenshots or []) if isinstance(p, str) and "vpage_" in p]
            dynamic_prompt = LLM1(image_paths=vpages)
            return original_build_llm_batch_payload(screenshots, prompt=dynamic_prompt)

        ha.build_llm_batch_payload = _build_llm_batch_payload_with_images
    except Exception:
        pass
    ensure_adb_running()

    cfg = DEFAULT_CONFIG
    cfg.verbose_logging = False
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
    long_score = long_score_result.get("score", 0) if isinstance(long_score_result, dict) else 0
    short_score = short_score_result.get("score", 0) if isinstance(short_score_result, dict) else 0

    T_LONG = 15
    T_SHORT = 20
    DOM_MARGIN = 10

    long_ok = long_score >= T_LONG
    short_ok = short_score >= T_SHORT
    long_delta = long_score - T_LONG
    short_delta = short_score - T_SHORT

    if not long_ok and not short_ok:
        decision = "reject"
    elif long_ok and (not short_ok or long_delta >= short_delta + DOM_MARGIN):
        decision = "long_pickup"
    elif short_ok and (not long_ok or short_delta >= long_delta + DOM_MARGIN):
        decision = "short_pickup"
    else:
        decision = "long_pickup"  # tie-break: prefer long

    # Dating intention hard routing overrides.
    dating_intention = _norm_value((_get_core(extracted) or {}).get("Dating Intentions", ""))
    if dating_intention in {
        _norm_value("Short-term relationship"),
    }:
        if decision == "long_pickup":
            decision = "reject"
    elif dating_intention == _norm_value("Life partner"):
        if decision == "short_pickup":
            decision = "reject"

    # Manual override (TEMP; remove when done)
    manual_override = ""
    try:
        print(
            "Gate decision pre-override: {decision} (long_score={long_score}, short_score={short_score}, "
            "long_delta={long_delta}, short_delta={short_delta})".format(
                decision=decision,
                long_score=long_score,
                short_score=short_score,
                long_delta=long_score - T_LONG,
                short_delta=short_score - T_SHORT,
            )
        )
        override = input("Override decision? (long/short/reject, blank to keep): ").strip().lower()
        if override in {"long", "short", "reject"}:
            manual_override = override
            decision = {"long": "long_pickup", "short": "short_pickup", "reject": "reject"}[override]
    except Exception:
        pass
    print(
        "GATE decision={decision} long_score={long_score} short_score={short_score} "
        "long_delta={long_delta} short_delta={short_delta} dom_margin={dom_margin}".format(
            decision=decision,
            long_score=long_score,
            short_score=short_score,
            long_delta=long_score - T_LONG,
            short_delta=short_score - T_SHORT,
            dom_margin=DOM_MARGIN,
        )
    )
    llm3_variant = ""
    llm3_result = {}
    llm4_result = {}
    target_action = {}
    if decision == "short_pickup":
        llm3_variant = "short"
        llm3_result = run_llm3_short(extracted)
    elif decision == "long_pickup":
        llm3_variant = "long"
        llm3_result = run_llm3_long(extracted)
    if llm3_result:
        llm4_result = run_llm4(llm3_result)
        target_id = str(llm4_result.get("main_target_id", "") or "").strip()
        if target_id:
            print(f"[TARGET] LLM4 chose target_id={target_id}")
            images_paths = llm_meta.get("images_paths", []) or []
            target_info = _resolve_target_from_extracted(extracted, target_id, images_paths)
            target_action = {"target_id": target_id, **target_info}
            print(
                "[TARGET] resolved type={type} source_file={source_file} page_half={page_half}".format(
                    type=target_info.get("type"),
                    source_file=target_info.get("source_file"),
                    page_half=target_info.get("page_half"),
                )
            )
            device = s.get("device")
            width = s.get("width")
            height = s.get("height")
            if target_info.get("source_file") and device and width and height:
                matched = _seek_to_source_file(
                    device,
                    width,
                    height,
                    target_info.get("source_file", ""),
                    target_info.get("page_half", ""),
                    max_swipes=12,
                    direction="up",
                )
                target_action["matched_screenshot"] = matched
                print(f"[TARGET] match={'yes' if matched else 'no'}")
                if matched:
                    tapped = _click_like_for_target(
                        device,
                        width,
                        height,
                        target_info.get("page_half", ""),
                    )
                    target_action["tap_like"] = bool(tapped)
                    print(f"[TARGET] tap_like={'yes' if tapped else 'no'}")
            else:
                if not target_info.get("source_file"):
                    print("[TARGET] no source_file resolved; skipping seek/tap")
                else:
                    print("[TARGET] device/size missing; skipping seek/tap")
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
        "gate_decision": decision,
        "gate_metrics": {
            "long_score": int(long_score),
            "short_score": int(short_score),
            "long_delta": int(long_score - T_LONG),
            "short_delta": int(short_score - T_SHORT),
            "dom_margin": int(DOM_MARGIN),
            "t_long": int(T_LONG),
            "t_short": int(T_SHORT),
        },
        "manual_override": manual_override,
        "llm3_variant": llm3_variant,
        "llm3_result": llm3_result,
        "llm4_result": llm4_result,
        "target_action": target_action,
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
    # Commented out for now to avoid interactive prompts during test runs.
    # try:
    #     name = (
    #         (extracted.get("Core Biometrics (Objective)", {}) or {}).get("Name")
    #         if isinstance(extracted, dict)
    #         else ""
    #     )
    #     long_score = long_score_result.get("score", 0) if isinstance(long_score_result, dict) else 0
    #     short_score = short_score_result.get("score", 0) if isinstance(short_score_result, dict) else 0
    #     print("\n=== Manual Preference Input (TEMP) ===")
    #     def _prompt_yes_no(label: str) -> str:
    #         while True:
    #             resp = input(label).strip().lower()
    #             if resp in {"y", "n"}:
    #                 return resp
    #             print("Please enter y or n.")
    #
    #     def _prompt_short_long(label: str) -> str:
    #         while True:
    #             resp = input(label).strip().lower()
    #             if resp in {"short"}:
    #                 return "short"
    #             if resp in {"long"}:
    #                 return "long"
    #             print("Please enter exactly 'short' or 'long'.")
    #
    #     user_like = _prompt_yes_no("Your verdict? (y/n): ")
    #     user_score_raw = input("Your score (0-10): ").strip()
    #     user_score = None
    #     try:
    #         user_score = int(user_score_raw)
    #     except Exception:
    #         user_score = None
    #     short_long = "n/a"
    #     if user_like == "y":
    #         short_long = _prompt_short_long("Short or long? (type exactly 'short' or 'long'): ")
    #     thoughts = input("Any thoughts? (optional): ").strip()
    #
    #     long_contributions = long_score_result.get("contributions", []) if isinstance(long_score_result, dict) else []
    #     short_contributions = short_score_result.get("contributions", []) if isinstance(short_score_result, dict) else []
    #     age = None
    #     if isinstance(long_score_result, dict):
    #         age = (long_score_result.get("signals", {}) or {}).get("declared_age")
    #     if age is None and isinstance(extracted, dict):
    #         age = (extracted.get("Core Biometrics (Objective)", {}) or {}).get("Age")
    #
    #     manual_log = {
    #         "timestamp": datetime.now().isoformat(timespec="seconds"),
    #         "name": name or "",
    #         "json_path": out_path or "",
    #         "long_score": int(long_score),
    #         "short_score": int(short_score),
    #         "gate_decision": decision,
    #         "gate_long_delta": int(long_score - T_LONG),
    #         "gate_short_delta": int(short_score - T_SHORT),
    #         "gate_dom_margin": int(DOM_MARGIN),
    #         "gate_t_long": int(T_LONG),
    #         "gate_t_short": int(T_SHORT),
    #         "user_like": user_like,
    #         "user_score": user_score,
    #         "short_long": short_long,
    #         "age": age,
    #         "long_score_contributions": long_contributions,
    #         "short_score_contributions": short_contributions,
    #         "thoughts": thoughts,
    #     }
    #     manual_log_path = os.path.join("logs", "manual_preference_log.jsonl")
    #     with open(manual_log_path, "a", encoding="utf-8") as f:
    #         f.write(json.dumps(manual_log, ensure_ascii=False) + "\n")
    #     print(f"Logged manual preferences to {manual_log_path}")
    # except Exception as e:
    #     print(f"Failed to log manual preferences: {e}")

    # Final concise decision summary (print last)
    try:
        def _top_contribs(score_result: Dict[str, Any], n: int = 3) -> str:
            contribs = score_result.get("contributions", []) if isinstance(score_result, dict) else []
            items = [
                (abs(int(c.get("delta", 0) or 0)), c)
                for c in contribs
                if int(c.get("delta", 0) or 0) != 0
            ]
            items.sort(key=lambda x: x[0], reverse=True)
            parts = []
            for _, c in items[:n]:
                parts.append(f"{c.get('field','')}: {c.get('value','')} ({c.get('delta','')})")
            return "; ".join(parts) if parts else "none"

        chosen_result = long_score_result if decision == "long_pickup" else short_score_result
        chosen_label = "long" if decision == "long_pickup" else ("short" if decision == "short_pickup" else "n/a")
        summary = (
            f"FINAL decision={decision} "
            f"long_score={long_score} short_score={short_score} "
            f"long_delta={long_score - T_LONG} short_delta={short_score - T_SHORT} "
            f"key_{chosen_label}_contributors={_top_contribs(chosen_result)}"
        )
        print(summary)
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
