#!/usr/bin/env python3
"""
Run a single full-scroll scrape + parse using a custom prompt.
Hardwired to Gemini and prints/saves the parsed JSON for easy review.
"""

import json
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Set, Tuple
from textwrap import dedent

from PIL import Image

import config  # ensure .env is loaded early via config.py

import hinge_agent as ha
from batch_payload import build_llm_batch_payload
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


def LLM1_VISUAL() -> str:
    """
    Visual-only prompt for LLM1: describe photos + infer visual traits.
    Images are provided in order: photo_1 ... photo_6.
    """
    return (
        "You are analyzing cropped photos from a Hinge profile. "
        "The images are provided in order: photo_1, photo_2, photo_3, photo_4, photo_5, photo_6.\n"
        "If fewer than 6 images are provided, leave missing descriptions empty.\n\n"
        "Return exactly one JSON object:\n\n"
        "{\n"
        '  "photos": [\n'
        '    {"id": "photo_1", "description": ""},\n'
        '    {"id": "photo_2", "description": ""},\n'
        '    {"id": "photo_3", "description": ""},\n'
        '    {"id": "photo_4", "description": ""},\n'
        '    {"id": "photo_5", "description": ""},\n'
        '    {"id": "photo_6", "description": ""}\n'
        "  ],\n"
        '  "visual_traits": {\n'
        '    "Face Visibility Quality": "",\n'
        '    "Photo Authenticity / Editing Level": "",\n'
        '    "Apparent Body Fat Level": "",\n'
        '    "Profile Distinctiveness": "",\n'
        '    "Apparent Build Category": "",\n'
        '    "Apparent Skin Tone": "",\n'
        '    "Apparent Ethnic Features": "",\n'
        '    "Hair Color": "",\n'
        '    "Facial Symmetry Level": "",\n'
        '    "Indicators of Fitness or Lifestyle": "",\n'
        '    "Overall Visual Appeal Vibe": "",\n'
        '    "Apparent Age Range Category": "",\n'
        '    "Attire and Style Indicators": "",\n'
        '    "Body Language and Expression": "",\n'
        '    "Visible Enhancements or Features": "",\n'
        '    "Apparent Upper Body Proportions": "",\n'
        '    "Apparent Attractiveness Tier": "",\n'
        '    "Reasoning for attractiveness tier": "",\n'
        '    "Facial Proportion Balance": "",\n'
        '    "Grooming Effort Level": "",\n'
        '    "Presentation Red Flags": "",\n'
        '    "Visible Tattoo Level": "",\n'
        '    "Visible Piercing Level": "",\n'
        '    "Short-Term / Hookup Orientation Signals": ""\n'
        "  }\n"
        "}\n\n"
        "Rules:\n"
        "- Return only the JSON object, nothing else.\n"
        "- Base everything ONLY on the photos (no text or profile info).\n"
        "- If something is unclear, leave the field empty.\n\n"
        "Photo description rules:\n"
        "- Provide a detailed, neutral visual summary of the main subject: clothing, pose, activity, background, "
        "facial features when visible, skin tone, build, grooming, accessories, and overall presentation. "
        "Avoid judgmental language.\n\n"
        "Visual traits allowed values (select exactly one unless it says multiple):\n\n"
        '"Face Visibility Quality": "Clear face in 3+ photos", "Clear face in 1-2 photos", "Face often partially obscured", "Face mostly not visible"\n'
        '"Photo Authenticity / Editing Level": "No obvious filters", "Some filters or mild editing", "Heavy filters/face smoothing", "Unclear"\n'
        '"Apparent Body Fat Level": "Low", "Average", "High", "Very high", "Unclear"\n'
        '"Profile Distinctiveness": "High (specific/unique)", "Medium", "Low (generic/boilerplate)", "Unclear"\n'
        '"Apparent Build Category": "Very slender/petite", "Slender/lean", "Athletic/toned/fit", "Average build", "Curvy (defined waist)", "Curvy (softer proportions)", "Heavy-set/stocky", "Obese/high body fat", "Muscular/built"\n'
        '"Apparent Skin Tone": "Very light/pale/fair", "Light/beige", "Warm light/tan", "Olive/medium-tan", "Golden/medium-brown", "Warm brown/deep tan", "Dark-brown/chestnut", "Very dark/ebony/deep"\n'
        '"Apparent Ethnic Features": "Ambiguous/unclear", "East Asian-presenting", "Southeast Asian-presenting", "South Asian-presenting", "Indian-presenting", "Jewish/Israeli-presenting", "Arab-presenting", "North African-presenting", "Middle Eastern-presenting (other/unspecified)", "Black/African-presenting", "Latino-presenting", "Nordic/Scandinavian-presenting", "Slavic/Eastern European-presenting", "Mediterranean/Southern European-presenting", "Western/Central European-presenting", "British/Irish-presenting", "White/European-presenting (unspecified)", "Mixed/ambiguous"\n'
        '"Hair Color": "Black", "Dark brown", "Medium brown", "Light brown", "Blonde", "Platinum blonde", "Red/ginger", "Gray/white", "Bald/shaved", "Dyed pink", "Dyed blue", "Dyed (unnatural other)", "Dyed (mixed/multiple colors)"\n'
        '"Facial Symmetry Level": "Very high", "High", "Moderate", "Low"\n'
        '"Indicators of Fitness or Lifestyle": "Visible muscle tone", "Athletic poses", "Sporty/athletic clothing", "Outdoor/active settings", "Gym/fitness context visible", "Sedentary/lounging poses", "No visible fitness indicators"\n'
        '"Overall Visual Appeal Vibe": "Very low-key/understated", "Natural/effortless", "Polished/elegant", "High-energy/adventurous", "Playful/flirty", "Sensual/alluring", "Edgy/alternative"\n'
        '"Apparent Age Range Category": "Late teens/early 20s (18-22)", "Early-mid 20s (23-26)", "Mid-late 20s (27-29)", "Early 30s (30-33)", "Mid 30s (34-37)", "Late 30s/early 40s (38-42)", "Mid 40s+ (43+)"\n'
        '"Attire and Style Indicators": "Very modest/covered", "Casual/comfortable", "Low-key/natural", "Polished/elegant", "Sporty/active", "Form-fitting/suggestive", "Highly revealing", "Edgy/alternative"\n'
        '"Body Language and Expression": "Shy/reserved", "Relaxed/casual", "Approachable/open", "Confident/engaging", "Playful/flirty", "Energetic/vibrant"\n'
        '"Visible Enhancements or Features": "None visible", "Glasses", "Sunglasses", "Makeup (light)", "Makeup (heavy)", "Jewelry", "Painted nails", "Very long nails (2cm+)", "Hair extensions/wig (obvious)", "False eyelashes (obvious)", "Hat/cap/beanie (worn in most photos)"\n'
        '"Apparent Upper Body Proportions": "Petite/small/narrow", "Average/balanced/proportional", "Defined/toned", "Full/curvy", "Prominent/voluptuous", "Broad/strong"\n'
        '"Apparent Attractiveness Tier": "Very unattractive/morbidly obese", "Low", "Average", "Above average", "High", "Very attractive", "Extremely attractive", "Supermodel"\n'
        '"Facial Proportion Balance": "Balanced/proportional", "Slightly unbalanced", "Noticeably unbalanced"\n'
        '"Grooming Effort Level": "Minimal/natural", "Moderate/casual", "High/polished", "Heavy/overdone"\n'
        '"Presentation Red Flags": "None", "Poor lighting", "Blurry/low resolution", "Unflattering angle", "Heavy filters/face smoothing", "Sunglasses in most photos", "Face mostly obscured", "Group photos unclear who is the subject", "Too many distant shots", "Mirror selfie cluttered", "Messy background", "Only one clear solo photo", "Awkward cropping", "Overexposed/washed out", "Inconsistent appearance across photos"\n'
        '"Visible Tattoo Level": "None visible", "Small/minimal", "Moderate", "High"\n'
        '"Visible Piercing Level": "None visible", "Minimal", "Moderate", "High"\n'
        '"Short-Term / Hookup Orientation Signals": "None evident", "Low", "Moderate", "High"\n'
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
        "Temp: You may only pick from photo ones, do not use prompt/polls \n\n"
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


def run_llm1_visual(
    image_paths: List[str],
    model: str | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    prompt = LLM1_VISUAL()
    payload = build_llm_batch_payload(image_paths, prompt=prompt)
    requested_model = model or get_default_model()
    resolved_model = resolve_model(requested_model)
    try:
        t0 = time.perf_counter()
        resp = get_llm_client().chat.completions.create(
            model=resolved_model,
            response_format={"type": "json_object"},
            messages=payload.get("messages", []),
        )
        _ = int((time.perf_counter() - t0) * 1000)
        raw = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            parsed = {}
        return parsed, payload.get("meta", {})
    except Exception as e:
        _log(f"[LLM1] visual call failed: {e}")
        return {}, payload.get("meta", {})


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


def _compute_ahash(img: Image.Image, size: int = 8) -> int:
    if img.mode != "L":
        img = img.convert("L")
    resample = getattr(Image, "LANCZOS", 1)
    small = img.resize((size, size), resample)
    pixels = list(small.getdata())
    if not pixels:
        return 0
    avg = sum(pixels) / len(pixels)
    bits = 0
    for i, p in enumerate(pixels):
        if p >= avg:
            bits |= 1 << i
    return bits


def _compute_center_ahash(
    img: Image.Image,
    size: int = 8,
    crop_ratio: float = 0.6,
) -> int:
    """
    Compute aHash on a center crop to reduce UI overlay influence (e.g., like button).
    """
    try:
        w, h = img.size
        side = int(min(w, h) * crop_ratio)
        if side <= 0:
            return _compute_ahash(img, size=size)
        left = max(0, (w - side) // 2)
        top = max(0, (h - side) // 2)
        crop = img.crop((left, top, left + side, top + side))
        return _compute_ahash(crop, size=size)
    except Exception:
        return _compute_ahash(img, size=size)


def _ahash_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _compute_ahash_from_file(path: str) -> Optional[int]:
    if not path or not os.path.isfile(path):
        return None
    try:
        img = Image.open(path).convert("RGB")
        return _compute_ahash(img)
    except Exception:
        return None


def _compute_center_ahash_from_file(
    path: str,
    crop_ratio: float = 0.6,
) -> Optional[int]:
    if not path or not os.path.isfile(path):
        return None
    try:
        img = Image.open(path).convert("RGB")
        return _compute_center_ahash(img, crop_ratio=crop_ratio)
    except Exception:
        return None


def _target_log(message: str) -> None:
    if os.getenv("HINGE_TARGET_DEBUG", "1") == "1":
        print(message)


def _log(message: str) -> None:
    # Always-on UI/debug logging for this rework (real-time).
    print(message, flush=True)


def _is_run_json_enabled() -> bool:
    return os.getenv("HINGE_SHOW_RUN_JSON", "0") == "1"


def _extract_xml_root(raw: str) -> str:
    """
    UIAutomator dumps sometimes include prefix text. Strip to the <hierarchy> root.
    """
    if not raw:
        return ""
    idx = raw.find("<hierarchy")
    if idx == -1:
        return ""
    return raw[idx:]


def _dump_ui_xml(device, tmp_path: str = "/sdcard/hinge_ui.xml") -> str:
    """
    Dump UI hierarchy to a temp path on-device and return the XML string.
    Uses a single rotating file to avoid cluttering the device storage.
    """
    try:
        device.shell(f"uiautomator dump {tmp_path}")
        raw = device.shell(f"cat {tmp_path}")
        # Best-effort cleanup; ignore failures.
        try:
            device.shell(f"rm {tmp_path}")
        except Exception:
            pass
        xml = _extract_xml_root(raw)
        if not xml:
            _log("[UI] Empty/invalid XML dump")
        return xml
    except Exception as e:
        _log(f"[UI] XML dump failed: {e}")
        return ""


def _parse_bounds(bounds: str) -> Optional[Tuple[int, int, int, int]]:
    if not bounds:
        return None
    try:
        left_top, right_bottom = bounds.split("][")
        left_top = left_top.replace("[", "")
        right_bottom = right_bottom.replace("]", "")
        x1, y1 = [int(v) for v in left_top.split(",")]
        x2, y2 = [int(v) for v in right_bottom.split(",")]
        return x1, y1, x2, y2
    except Exception:
        return None


def _bounds_center(bounds: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bounds
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def _bounds_close(
    a: Optional[Tuple[int, int, int, int]],
    b: Optional[Tuple[int, int, int, int]],
    tol: int = 24,
) -> bool:
    if not a or not b:
        return False
    ax, ay = _bounds_center(a)
    bx, by = _bounds_center(b)
    return abs(ax - bx) <= tol and abs(ay - by) <= tol


def _parse_prompt_content_desc(content_desc: str) -> Tuple[str, str]:
    cd = (content_desc or "").strip()
    if not cd.startswith("Prompt:"):
        return "", ""
    if "Answer:" not in cd:
        return "", ""
    prompt_part, answer_part = cd.split("Answer:", 1)
    prompt_text = prompt_part.replace("Prompt:", "").strip().strip(".")
    answer_text = answer_part.strip()
    return prompt_text, answer_text


def _find_prompt_bounds_by_text(
    nodes: List[Dict[str, Any]],
    prompt_text: str,
    answer_text: str,
) -> Optional[Tuple[int, int, int, int]]:
    if not prompt_text or not answer_text:
        return None
    target_key = _normalize_text_basic(prompt_text) + "||" + _normalize_text_basic(answer_text)
    for n in nodes:
        cd = (n.get("content_desc") or "").strip()
        if not cd.startswith("Prompt:"):
            continue
        p_txt, a_txt = _parse_prompt_content_desc(cd)
        if not p_txt or not a_txt:
            continue
        key = _normalize_text_basic(p_txt) + "||" + _normalize_text_basic(a_txt)
        if key == target_key:
            return n.get("bounds")
    return None


def _find_poll_option_bounds_by_text(
    nodes: List[Dict[str, Any]],
    option_text: str,
) -> Optional[Tuple[int, int, int, int]]:
    if not option_text:
        return None
    target_norm = _normalize_text_basic(option_text)
    for n in nodes:
        cd = (n.get("content_desc") or "").strip()
        if not cd.startswith("Option:"):
            continue
        opt_text = cd.replace("Option:", "").strip()
        if _normalize_text_basic(opt_text) == target_norm:
            return n.get("bounds")
    return None


def _find_like_button_near_bounds_screen(
    nodes: List[Dict[str, Any]],
    target_bounds: Tuple[int, int, int, int],
    prefer_type: str,
    max_gap: int = 160,
) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
    prefer = (prefer_type or "").lower()
    tb = target_bounds
    candidates: List[Tuple[int, Tuple[int, int, int, int], str]] = []
    fallback: List[Tuple[int, Tuple[int, int, int, int], str]] = []
    for n in nodes:
        if n.get("cls") != "android.widget.Button":
            continue
        cd = (n.get("content_desc") or "").strip()
        if "like" not in cd.lower():
            continue
        b = n.get("bounds")
        if not b:
            continue
        ly = _bounds_center(b)[1]
        if tb[1] <= ly <= tb[3] + max_gap:
            dist = 0 if tb[1] <= ly <= tb[3] else abs(ly - tb[3])
        else:
            dist = abs(ly - tb[3])
        entry = (dist, b, cd)
        if prefer and prefer in cd.lower():
            candidates.append(entry)
        else:
            fallback.append(entry)
    pool = candidates if candidates else fallback
    if not pool:
        return None, ""
    pool.sort(key=lambda x: x[0])
    if pool[0][0] > max_gap:
        return None, ""
    _, best_b, best_cd = pool[0]
    return best_b, best_cd


def _flatten_ui_nodes(root: ET.Element) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []

    def walk(el: ET.Element) -> None:
        attrs = el.attrib or {}
        bounds = _parse_bounds(attrs.get("bounds", ""))
        node = {
            "text": attrs.get("text", "") or "",
            "content_desc": attrs.get("content-desc", "") or "",
            "cls": attrs.get("class", "") or "",
            "scrollable": attrs.get("scrollable", "") == "true",
            "bounds": bounds,
        }
        if bounds:
            nodes.append(node)
        for child in list(el):
            walk(child)

    walk(root)
    return nodes


def _parse_ui_nodes(xml_text: str) -> List[Dict[str, Any]]:
    if not xml_text:
        return []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []
    return _flatten_ui_nodes(root)


def _find_scroll_area(nodes: List[Dict[str, Any]]) -> Optional[Tuple[int, int, int, int]]:
    # Choose the largest scrollable container (by height) as the profile scroll area.
    scroll_nodes = [n for n in nodes if n.get("scrollable") and n.get("bounds")]
    if not scroll_nodes:
        return None
    return max(scroll_nodes, key=lambda n: n["bounds"][3] - n["bounds"][1])["bounds"]


def _find_horizontal_scroll_area(
    nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
) -> Optional[Tuple[int, int, int, int]]:
    """
    Heuristic to find the horizontal biometrics scroller inside the profile.
    """
    top, bottom = scroll_area[1], scroll_area[3]
    candidates: List[Tuple[int, Tuple[int, int, int, int]]] = []
    for n in nodes:
        if not n.get("scrollable"):
            continue
        b = n.get("bounds")
        if not b:
            continue
        # Must be inside the vertical scroll area.
        if b[3] <= top or b[1] >= bottom:
            continue
        w = b[2] - b[0]
        h = b[3] - b[1]
        if h <= 0:
            continue
        # Wide + short = likely horizontal scroller.
        if w >= 600 and (w / h) >= 2.0 and h <= 260:
            candidates.append((w, b))
    if not candidates:
        return None
    # Prefer the widest candidate.
    return sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]


def _node_key(node: Dict[str, Any]) -> str:
    # Use content-desc or text to match nodes across scrolls.
    cd = (node.get("content_desc") or "").strip()
    tx = (node.get("text") or "").strip()
    if cd:
        return f"cd:{cd}"
    if tx:
        return f"tx:{tx}"
    return ""


def _compute_scroll_delta(
    prev_nodes: List[Dict[str, Any]],
    curr_nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
) -> Optional[int]:
    """
    Estimate scroll delta by matching stable nodes between two XML dumps.
    Returns positive pixels when scrolling DOWN (content moves up).
    """
    if not prev_nodes or not curr_nodes:
        return None
    top, bottom = scroll_area[1], scroll_area[3]

    def in_scroll(n: Dict[str, Any]) -> bool:
        b = n.get("bounds")
        if not b:
            return False
        return b[1] < bottom and b[3] > top

    prev_map: Dict[str, List[int]] = {}
    for n in prev_nodes:
        if not in_scroll(n):
            continue
        key = _node_key(n)
        if not key or key.startswith("cd:Like"):
            continue
        _, y = _bounds_center(n["bounds"])
        prev_map.setdefault(key, []).append(y)

    deltas: List[int] = []
    for n in curr_nodes:
        if not in_scroll(n):
            continue
        key = _node_key(n)
        if not key or key.startswith("cd:Like"):
            continue
        if key not in prev_map:
            continue
        _, y = _bounds_center(n["bounds"])
        # Match against the closest prior y for this key.
        prev_ys = prev_map.get(key, [])
        if not prev_ys:
            continue
        closest_prev = min(prev_ys, key=lambda py: abs(py - y))
        deltas.append(closest_prev - y)

    if not deltas:
        return None
    deltas.sort()
    return deltas[len(deltas) // 2]


def _screen_signature(
    nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
) -> Set[Tuple[str, int]]:
    """
    Build a coarse signature of visible nodes to detect no-move scrolls.
    """
    top, bottom = scroll_area[1], scroll_area[3]
    sig: Set[Tuple[str, int]] = set()
    for n in nodes:
        b = n.get("bounds")
        if not b:
            continue
        if not (b[1] < bottom and b[3] > top):
            continue
        key = _node_key(n)
        if not key or key.startswith("cd:Like"):
            continue
        _, cy = _bounds_center(b)
        sig.add((key, int(round(cy / 10.0)) * 10))
    return sig


def _annotate_nodes_with_abs_bounds(
    nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
    scroll_offset: int,
) -> List[Dict[str, Any]]:
    """
    Convert on-screen bounds to absolute content bounds by adding scroll_offset
    for nodes that fall inside the scrollable profile area.
    """
    top, bottom = scroll_area[1], scroll_area[3]
    annotated: List[Dict[str, Any]] = []
    for n in nodes:
        b = n.get("bounds")
        if not b:
            continue
        in_scroll = b[1] < bottom and b[3] > top
        abs_bounds = (b[0], b[1] + scroll_offset, b[2], b[3] + scroll_offset) if in_scroll else b
        nn = dict(n)
        nn["in_scroll"] = in_scroll
        nn["abs_bounds"] = abs_bounds
        annotated.append(nn)
    return annotated


_BIOMETRIC_LABEL_MAP = {
    "age": "Age",
    "gender": "Gender",
    "sexuality": "Sexuality",
    "height": "Height",
    "job": "Job title",
    "job title": "Job title",
    "college or university": "University",
    "university": "University",
    "religion": "Religious Beliefs",
    "home town": "Home town",
    "hometown": "Home town",
    "languages spoken": "Languages spoken",
    "ethnicity": "Explicit Ethnicity",
    "dating intentions": "Dating Intentions",
    "relationship type": "Relationship type",
    "children": "Children",
    "family plans": "Family plans",
    "covid vaccine": "Covid Vaccine",
    "politics": "Politics",
    "zodiac sign": "Zodiac Sign",
    "pets": "Pets",
    "drinking": "Drinking",
    "smoking": "Smoking",
    "marijuana": "Marijuana",
    "drugs": "Drugs",
    "location": "Location",
}


def _normalize_label(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _parse_height_value(raw: str) -> Optional[int]:
    if not raw:
        return None
    s = raw.strip().lower()
    # If explicitly in cm (or looks like cm), take first number.
    import re
    nums = [int(n) for n in re.findall(r"\d+", s)]
    if not nums:
        return None
    if "cm" in s or (nums and nums[0] >= 100):
        return nums[0]
    # Handle feet/inches formats like 5'7 or 5 ft 7 in.
    if "'" in s or "ft" in s:
        feet = nums[0] if nums else 0
        inches = nums[1] if len(nums) > 1 else 0
        cm = int(round(feet * 30.48 + inches * 2.54))
        return cm
    # Fallback: if two small numbers, treat as feet/inches.
    if len(nums) >= 2 and nums[0] <= 7 and nums[1] <= 11:
        cm = int(round(nums[0] * 30.48 + nums[1] * 2.54))
        return cm
    return nums[0]


def _extract_biometrics_from_nodes(
    nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
) -> Dict[str, Any]:
    """
    Extract biometrics visible on the current screen by pairing label nodes
    (content-desc) with value text nodes to their right.
    """
    top, bottom = scroll_area[1], scroll_area[3]

    def in_scroll(n: Dict[str, Any]) -> bool:
        b = n.get("bounds")
        if not b:
            return False
        return b[1] < bottom and b[3] > top

    label_nodes: List[Dict[str, Any]] = []
    value_nodes: List[Dict[str, Any]] = []
    for n in nodes:
        if not in_scroll(n):
            continue
        cd = n.get("content_desc") or ""
        tx = n.get("text") or ""
        if cd:
            label_nodes.append(n)
        if tx:
            value_nodes.append(n)

    updates: Dict[str, Any] = {}
    for label in label_nodes:
        label_text = _normalize_label(label.get("content_desc", ""))
        if label_text not in _BIOMETRIC_LABEL_MAP:
            continue
        field = _BIOMETRIC_LABEL_MAP[label_text]
        lb = label["bounds"]
        _, ly = _bounds_center(lb)
        best_val = None
        best_score = None
        for val in value_nodes:
            vb = val["bounds"]
            # Value should be to the right of the label and roughly aligned vertically.
            if vb[0] < lb[2] - 5:
                continue
            _, vy = _bounds_center(vb)
            if abs(vy - ly) > max(60, (lb[3] - lb[1]) * 1.5):
                continue
            dx = vb[0] - lb[2]
            score = abs(vy - ly) * 2 + dx
            if best_score is None or score < best_score:
                best_score = score
                best_val = val
        if not best_val:
            continue
        raw_val = (best_val.get("text") or "").strip()
        if not raw_val:
            continue
        if field == "Age":
            try:
                updates[field] = int("".join(ch for ch in raw_val if ch.isdigit()))
            except Exception:
                continue
        elif field == "Height":
            height_val = _parse_height_value(raw_val)
            if height_val is not None:
                updates[field] = height_val
        else:
            updates[field] = raw_val
    return updates


def _hscroll_once(
    device,
    area: Tuple[int, int, int, int],
    direction: str = "left",
    distance_px: Optional[int] = None,
    duration_ms: int = 350,
) -> None:
    left, top, right, bottom = area
    mid_y = int((top + bottom) / 2)
    area_w = right - left
    dist = int(distance_px or (area_w * 0.6))
    dist = max(80, min(dist, int(area_w * 0.9)))
    if direction == "left":
        x_start = int(right - area_w * 0.1)
        x_end = max(int(left + area_w * 0.1), x_start - dist)
    else:
        x_start = int(left + area_w * 0.1)
        x_end = min(int(right - area_w * 0.1), x_start + dist)
    swipe(device, x_start, mid_y, x_end, mid_y, duration_ms)
    _log(f"[BIOMETRICS] hscroll {direction} x={x_start}->{x_end} y={mid_y}")


def _scan_biometrics_hscroll(
    device,
    nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
    biometrics: Dict[str, Any],
    max_swipes: int = 12,
) -> List[Dict[str, Any]]:
    """
    Attempt to reveal additional biometrics by horizontal scrolling.
    """
    h_area = _find_horizontal_scroll_area(nodes, scroll_area)
    if not h_area:
        _log("[BIOMETRICS] no horizontal scroller detected")
        return nodes
    _log(f"[BIOMETRICS] horizontal scroller area={h_area}")
    swipes_done = 0
    no_new = 0
    while swipes_done < max_swipes and no_new < 2:
        _hscroll_once(device, h_area, "left")
        time.sleep(0.4)
        xml = _dump_ui_xml(device)
        nodes = _parse_ui_nodes(xml)
        updates = _extract_biometrics_from_nodes(nodes, scroll_area)
        new_any = False
        for k, v in updates.items():
            if k not in biometrics and v not in ("", None):
                biometrics[k] = v
                _log(f"[BIOMETRICS] {k} = {v}")
                new_any = True
        if new_any:
            no_new = 0
        else:
            no_new += 1
        swipes_done += 1
    return nodes


def _add_or_update_by_abs_y(
    items: List[Dict[str, Any]],
    new_item: Dict[str, Any],
    y_key: str = "abs_center_y",
    tol: int = 20,
    dedupe_key: Optional[str] = None,
) -> None:
    if dedupe_key:
        for item in items:
            if item.get(dedupe_key) == new_item.get(dedupe_key):
                # Update bounds if missing or more complete.
                if not item.get("abs_bounds") and new_item.get("abs_bounds"):
                    item["abs_bounds"] = new_item["abs_bounds"]
                return
    new_y = new_item.get(y_key)
    if new_y is None:
        items.append(new_item)
        return
    for item in items:
        if abs((item.get(y_key) or 0) - new_y) <= tol:
            # Prefer larger bounds (more complete photo).
            if new_item.get("abs_bounds") and item.get("abs_bounds"):
                ob = item["abs_bounds"]
                nb = new_item["abs_bounds"]
                if (nb[3] - nb[1]) > (ob[3] - ob[1]):
                    item["abs_bounds"] = nb
            return
    items.append(new_item)


def _nearest_like_bounds(
    card_bounds: Tuple[int, int, int, int],
    likes: List[Dict[str, Any]],
    max_gap: int = 180,
) -> Optional[Tuple[int, int, int, int]]:
    if not likes:
        return None
    cb = card_bounds
    candidates: List[Tuple[int, Tuple[int, int, int, int]]] = []
    for l in likes:
        lb = l.get("abs_bounds")
        if not lb:
            continue
        ly = _bounds_center(lb)[1]
        # Prefer likes vertically inside the card bounds (or just below).
        if cb[1] <= ly <= cb[3] + max_gap:
            dist = 0 if cb[1] <= ly <= cb[3] else abs(ly - cb[3])
        else:
            dist = abs(ly - cb[3])
        candidates.append((dist, lb))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    # Only accept if reasonably close.
    if candidates[0][0] > max_gap:
        return None
    return candidates[0][1]


def _update_ui_map(
    ui_map: Dict[str, Any],
    nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
    scroll_offset: int,
) -> None:
    """
    Update UI map with prompts, polls, photos, and like buttons for the current screen.
    """
    annotated = _annotate_nodes_with_abs_bounds(nodes, scroll_area, scroll_offset)
    top, bottom = scroll_area[1], scroll_area[3]

    def in_scroll(n: Dict[str, Any]) -> bool:
        b = n.get("bounds")
        if not b:
            return False
        return b[1] < bottom and b[3] > top

    # Collect like buttons visible on this screen first.
    screen_likes: List[Dict[str, Any]] = []
    for n in annotated:
        if not n.get("in_scroll"):
            continue
        cd = (n.get("content_desc") or "").strip()
        cls = n.get("cls") or ""
        b = n.get("abs_bounds")
        if not b:
            continue
        cx, cy = _bounds_center(b)

        if cls == "android.widget.Button" and cd.lower().startswith("like"):
            cd_lower = cd.lower()
            if "photo" in cd_lower:
                like_type = "photo"
            elif "prompt" in cd_lower:
                like_type = "prompt"
            else:
                like_type = "unknown"
            screen_likes.append(
                {
                    "type": like_type,
                    "abs_bounds": b,
                    "abs_center_y": cy,
                    "content_desc": cd,
                }
            )

    screen_photo_likes = [l for l in screen_likes if l.get("type") == "photo"]
    screen_prompt_likes = [l for l in screen_likes if l.get("type") == "prompt"]

    # Add likes to the global map.
    for l in screen_likes:
        _add_or_update_by_abs_y(
            ui_map["likes"],
            {
                "type": l["type"],
                "abs_bounds": l["abs_bounds"],
                "abs_center_y": l["abs_center_y"],
                "content_desc": l.get("content_desc", ""),
            },
            tol=10,
        )

    # Second pass for prompts/polls/photos.
    for n in annotated:
        if not n.get("in_scroll"):
            continue
        cd = (n.get("content_desc") or "").strip()
        cls = n.get("cls") or ""
        b = n.get("abs_bounds")
        if not b:
            continue
        cx, cy = _bounds_center(b)

        # Prompt cards and poll questions.
        if cd.startswith("Prompt:"):
            # If it contains Answer:, treat as prompt card.
            if "Answer:" in cd:
                prompt_part, answer_part = cd.split("Answer:", 1)
                prompt_text = prompt_part.replace("Prompt:", "").strip().strip(".")
                answer_text = answer_part.strip()
                key = f"{prompt_text}||{answer_text}"
                prompt_like = _nearest_like_bounds(b, screen_prompt_likes, max_gap=120)
                prompt_item = {
                    "prompt": prompt_text,
                    "answer": answer_text,
                    "abs_bounds": b,
                    "abs_center_y": cy,
                    "key": key,
                }
                if prompt_like:
                    prompt_item["like_bounds"] = prompt_like
                _add_or_update_by_abs_y(ui_map["prompts"], prompt_item, dedupe_key="key")
            else:
                # Poll question (prompt without Answer)
                question = cd.replace("Prompt:", "").strip()
                if question and not ui_map["poll"].get("question"):
                    ui_map["poll"]["question"] = question

        # Poll options
        if cd.startswith("Option:"):
            option_text = cd.replace("Option:", "").strip()
            if option_text:
                _add_or_update_by_abs_y(
                    ui_map["poll"]["options"],
                    {
                        "text": option_text,
                        "abs_bounds": b,
                        "abs_center_y": cy,
                        "key": option_text,
                    },
                    dedupe_key="key",
                )

        # Photos
        if cls == "android.widget.ImageView" and "photo" in cd.lower():
            width = b[2] - b[0]
            height = b[3] - b[1]
            if width >= 200 and height >= 200:
                photo_like = _nearest_like_bounds(b, screen_photo_likes, max_gap=180)
                like_key = None
                if photo_like:
                    like_key = ",".join(str(v) for v in photo_like)
                _add_or_update_by_abs_y(
                    ui_map["photos"],
                    {
                        "content_desc": cd,
                        "abs_bounds": b,
                        "abs_center_y": cy,
                        "width": width,
                        "height": height,
                        "like_bounds": photo_like,
                        "like_key": like_key,
                    },
                    tol=120,
                    dedupe_key="like_key" if like_key else None,
                )


def _update_ui_map_text_only(
    ui_map: Dict[str, Any],
    nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
    scroll_offset: int,
) -> None:
    """
    Update UI map with prompts, polls, and like buttons only (no photos).
    """
    annotated = _annotate_nodes_with_abs_bounds(nodes, scroll_area, scroll_offset)
    top, bottom = scroll_area[1], scroll_area[3]

    # Collect like buttons visible on this screen first.
    screen_likes: List[Dict[str, Any]] = []
    for n in annotated:
        if not n.get("in_scroll"):
            continue
        cd = (n.get("content_desc") or "").strip()
        cls = n.get("cls") or ""
        b = n.get("abs_bounds")
        if not b:
            continue
        _, cy = _bounds_center(b)
        if cls == "android.widget.Button" and cd.lower().startswith("like"):
            cd_lower = cd.lower()
            if "photo" in cd_lower:
                like_type = "photo"
            elif "prompt" in cd_lower:
                like_type = "prompt"
            else:
                like_type = "unknown"
            screen_likes.append(
                {
                    "type": like_type,
                    "abs_bounds": b,
                    "abs_center_y": cy,
                    "content_desc": cd,
                }
            )

    screen_prompt_likes = [l for l in screen_likes if l.get("type") == "prompt"]

    for l in screen_likes:
        _add_or_update_by_abs_y(
            ui_map["likes"],
            {
                "type": l["type"],
                "abs_bounds": l["abs_bounds"],
                "abs_center_y": l["abs_center_y"],
                "content_desc": l.get("content_desc", ""),
            },
            tol=10,
        )

    for n in annotated:
        if not n.get("in_scroll"):
            continue
        cd = (n.get("content_desc") or "").strip()
        b = n.get("abs_bounds")
        if not b:
            continue
        _, cy = _bounds_center(b)

        if cd.startswith("Prompt:"):
            if "Answer:" in cd:
                prompt_part, answer_part = cd.split("Answer:", 1)
                prompt_text = prompt_part.replace("Prompt:", "").strip().strip(".")
                answer_text = answer_part.strip()
                key = f"{prompt_text}||{answer_text}"
                prompt_like = _nearest_like_bounds(b, screen_prompt_likes, max_gap=120)
                prompt_item = {
                    "prompt": prompt_text,
                    "answer": answer_text,
                    "abs_bounds": b,
                    "abs_center_y": cy,
                    "key": key,
                }
                if prompt_like:
                    prompt_item["like_bounds"] = prompt_like
                _add_or_update_by_abs_y(ui_map["prompts"], prompt_item, dedupe_key="key")
            else:
                question = cd.replace("Prompt:", "").strip()
                if question and not ui_map["poll"].get("question"):
                    ui_map["poll"]["question"] = question

        if cd.startswith("Option:"):
            option_text = cd.replace("Option:", "").strip()
            if option_text:
                _add_or_update_by_abs_y(
                    ui_map["poll"]["options"],
                    {
                        "text": option_text,
                        "abs_bounds": b,
                        "abs_center_y": cy,
                        "key": option_text,
                    },
                    dedupe_key="key",
                )


def _find_primary_photo_bounds(
    nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
) -> Optional[Tuple[int, int, int, int]]:
    top, bottom = scroll_area[1], scroll_area[3]
    best = None
    best_area = None
    for n in nodes:
        if n.get("cls") != "android.widget.ImageView":
            continue
        cd = (n.get("content_desc") or "").lower()
        if "photo" not in cd:
            continue
        b = n.get("bounds")
        if not b:
            continue
        if b[1] >= bottom or b[3] <= top:
            continue
        w = b[2] - b[0]
        h = b[3] - b[1]
        if w < 200 or h < 200:
            continue
        area = w * h
        if best_area is None or area > best_area:
            best_area = area
            best = b
    return best


def _find_visible_photo_bounds_all(
    nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
) -> List[Tuple[int, int, int, int]]:
    top, bottom = scroll_area[1], scroll_area[3]
    results: List[Tuple[int, int, int, int]] = []
    for n in nodes:
        if n.get("cls") != "android.widget.ImageView":
            continue
        cd = (n.get("content_desc") or "").lower()
        if "photo" not in cd:
            continue
        b = n.get("bounds")
        if not b:
            continue
        if b[1] >= bottom or b[3] <= top:
            continue
        results.append(b)
    return results


def _clamp_bounds_to_screen(
    bounds: Tuple[int, int, int, int],
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = bounds
    x1 = max(0, min(width - 1, x1))
    x2 = max(1, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(1, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

def _is_square_bounds(bounds: Tuple[int, int, int, int], tol: int = 12) -> bool:
    w = bounds[2] - bounds[0]
    h = bounds[3] - bounds[1]
    return abs(w - h) <= tol


def _compute_center_ahash_from_bounds(
    device,
    bounds: Tuple[int, int, int, int],
    width: int,
    height: int,
    crop_ratio: float = 0.6,
) -> Optional[int]:
    cb = _clamp_bounds_to_screen(bounds, width, height)
    if not cb:
        return None
    try:
        img_bytes = device.screencap()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        crop = img.crop(cb)
        return _compute_center_ahash(crop, crop_ratio=crop_ratio)
    except Exception:
        return None


def _match_photo_bounds_by_hash(
    device,
    width: int,
    height: int,
    nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
    target_hash: int,
    expected_screen_y: Optional[int] = None,
    max_dist: int = 18,
    square_only: bool = True,
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[int]]:
    candidates = _find_visible_photo_bounds_all(nodes, scroll_area)
    if not candidates:
        return None, None
    if square_only:
        square_candidates = [b for b in candidates if _is_square_bounds(b)]
        if square_candidates:
            candidates = square_candidates
        else:
            _log("[TARGET] no square photo candidates; retrying with partials")
    if expected_screen_y is not None:
        candidates.sort(key=lambda b: abs(_bounds_center(b)[1] - expected_screen_y))
    img_bytes = device.screencap()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    best_bounds = None
    best_dist = None
    for b in candidates:
        cb = _clamp_bounds_to_screen(b, width, height)
        if not cb:
            continue
        crop = img.crop(cb)
        h = _compute_center_ahash(crop)
        dist = _ahash_distance(h, target_hash)
        _log(f"[TARGET] photo hash candidate bounds={cb} dist={dist}")
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_bounds = cb
    if best_dist is None:
        return None, None
    if best_dist <= max_dist:
        return best_bounds, best_dist
    return None, best_dist


def _find_like_button_in_photo(
    nodes: List[Dict[str, Any]],
    photo_bounds: Tuple[int, int, int, int],
) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
    x1, y1, x2, y2 = photo_bounds
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    best = None
    best_score = None
    best_desc = ""
    for n in nodes:
        if n.get("cls") != "android.widget.Button":
            continue
        cd = (n.get("content_desc") or "").strip()
        if "like" not in cd.lower():
            continue
        b = n.get("bounds")
        if not b:
            continue
        cx, cy = _bounds_center(b)
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            continue
        score = (cx - mid_x) + (cy - mid_y)
        if best_score is None or score > best_score:
            best_score = score
            best = b
            best_desc = cd
    if best:
        return best, best_desc

    # Fallback: nearest like button to bottom-right corner.
    br_x, br_y = x2, y2
    fallback = None
    fallback_desc = ""
    fallback_dist = None
    for n in nodes:
        if n.get("cls") != "android.widget.Button":
            continue
        cd = (n.get("content_desc") or "").strip()
        if "like" not in cd.lower():
            continue
        b = n.get("bounds")
        if not b:
            continue
        cx, cy = _bounds_center(b)
        dist = abs(cx - br_x) + abs(cy - br_y)
        if fallback_dist is None or dist < fallback_dist:
            fallback_dist = dist
            fallback = b
            fallback_desc = cd
    return fallback, fallback_desc


def _find_like_button_near_expected(
    nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
    target_type: str,
    expected_screen_y: int,
) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
    """
    Find the closest like button on screen to the expected Y.
    Prefer matching content-desc for prompt/photo when possible.
    """
    top, bottom = scroll_area[1], scroll_area[3]
    target_type = (target_type or "").strip().lower()
    prefer = ""
    if target_type == "photo":
        prefer = "photo"
    elif target_type == "prompt":
        prefer = "prompt"

    candidates: List[Tuple[int, Tuple[int, int, int, int], str]] = []
    fallback: List[Tuple[int, Tuple[int, int, int, int], str]] = []
    for n in nodes:
        if n.get("cls") != "android.widget.Button":
            continue
        cd = (n.get("content_desc") or "").strip()
        if "like" not in cd.lower():
            continue
        b = n.get("bounds")
        if not b:
            continue
        if b[1] >= bottom or b[3] <= top:
            continue
        cy = _bounds_center(b)[1]
        dist = abs(cy - expected_screen_y)
        entry = (dist, b, cd)
        if prefer and prefer in cd.lower():
            candidates.append(entry)
        else:
            fallback.append(entry)

    pool = candidates if candidates else fallback
    if not pool:
        return None, ""
    pool.sort(key=lambda x: x[0])
    _, best_b, best_cd = pool[0]
    return best_b, best_cd


def _ensure_photo_square(
    device,
    width: int,
    height: int,
    scroll_area: Tuple[int, int, int, int],
    nodes: List[Dict[str, Any]],
    offset: int,
    photo_bounds: Tuple[int, int, int, int],
    max_attempts: int = 4,
    target_abs_center_y: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], int, Optional[Tuple[int, int, int, int]]]:
    vb = photo_bounds
    vb_w = vb[2] - vb[0]
    vb_h = vb[3] - vb[1]
    is_square = _is_square_bounds(vb)
    attempts = 0
    while not is_square and attempts < max_attempts:
        top_clipped = vb[1] <= scroll_area[1] + 3
        bottom_clipped = vb[3] >= scroll_area[3] - 3
        direction = None
        if top_clipped:
            # At absolute top, we can't scroll up further; reveal bottom instead.
            direction = "down" if offset <= 0 else "up"
        elif bottom_clipped:
            direction = "down"
        if not direction:
            break
        prev_nodes = nodes
        nodes, delta = _scroll_and_capture(
            device,
            width,
            height,
            scroll_area,
            direction,
            prev_nodes,
            distance_px=140,
        )
        offset += delta
        if target_abs_center_y is not None:
            expected_screen_y = int(target_abs_center_y - offset)
            vb = _find_visible_photo_bounds(nodes, scroll_area, expected_screen_y)
        else:
            vb = _find_primary_photo_bounds(nodes, scroll_area)
        if not vb:
            break
        vb_w = vb[2] - vb[0]
        vb_h = vb[3] - vb[1]
        is_square = _is_square_bounds(vb)
        _log(
            f"[PHOTO] micro-scroll {attempts+1} dir={direction} size={vb_w}x{vb_h} square={'yes' if is_square else 'no'}"
        )
        attempts += 1
    return nodes, offset, vb

def _assign_like_buttons(ui_map: Dict[str, Any]) -> None:
    """
    Attach the nearest like button to each prompt/photo card.
    """
    likes = ui_map.get("likes", [])
    for l in likes:
        l["used"] = False

    def assign(cards: List[Dict[str, Any]], prefer_type: str) -> None:
        for card in cards:
            if card.get("like_bounds"):
                continue
            cb = card.get("abs_bounds")
            if not cb:
                continue
            cx, cy = _bounds_center(cb)
            candidates = [
                l for l in likes
                if not l.get("used") and l.get("abs_bounds")
                and l.get("type") == prefer_type
            ]
            if not candidates:
                continue
            def dist(like: Dict[str, Any]) -> int:
                ly = _bounds_center(like["abs_bounds"])[1]
                if cb[1] <= ly <= cb[3]:
                    return 0
                return min(abs(ly - cb[1]), abs(ly - cb[3]))
            best = min(candidates, key=dist)
            best["used"] = True
            card["like_bounds"] = best["abs_bounds"]
            card["like_desc"] = best.get("content_desc", "")

    assign(ui_map.get("prompts", []), "prompt")
    assign(ui_map.get("photos", []), "photo")


def _dedupe_photos_by_like_bounds(ui_map: Dict[str, Any]) -> None:
    """
    Log duplicate photo entries that resolve to the same like button bounds.
    """
    seen = set()
    for p in ui_map.get("photos", []):
        lb = p.get("like_bounds")
        if not lb:
            continue
        key = tuple(lb)
        if key in seen:
            _log(f"[MAP] duplicate photo like_bounds={key}")
        else:
            seen.add(key)


def _assign_ids(ui_map: Dict[str, Any]) -> None:
    # Sort top-to-bottom for stable IDs.
    ui_map["prompts"].sort(key=lambda p: p.get("abs_center_y", 0))
    for idx, prompt in enumerate(ui_map["prompts"], start=1):
        prompt["id"] = f"prompt_{idx}"

    ui_map["photos"].sort(key=lambda p: p.get("abs_center_y", 0))
    for idx, photo in enumerate(ui_map["photos"], start=1):
        photo["id"] = f"photo_{idx}"

    ui_map["poll"]["options"].sort(key=lambda o: o.get("abs_center_y", 0))
    for idx, opt in enumerate(ui_map["poll"]["options"], start=1):
        suffix = chr(ord("a") + idx - 1)
        opt["id"] = f"poll_1_{suffix}"


def _scroll_once(
    device,
    width: int,
    height: int,
    scroll_area: Tuple[int, int, int, int],
    direction: str = "down",
    distance_px: Optional[int] = None,
    duration_ms: int = 450,
) -> int:
    """
    Perform a single scroll gesture.
    direction="down" means moving down the profile (content moves up).
    Returns the expected scroll delta in pixels (positive for down, negative for up).
    """
    left, top, right, bottom = scroll_area
    area_h = bottom - top
    x = int((left + right) / 2)
    dist = int(distance_px or (area_h * 0.6))
    dist = max(80, min(dist, int(area_h * 0.9)))

    if direction == "down":
        # Finger swipes up.
        y_start = int(bottom - area_h * 0.15)
        y_end = max(int(top + area_h * 0.1), y_start - dist)
        swipe(device, x, y_start, x, y_end, duration_ms)
        expected = y_start - y_end
    else:
        # direction == "up": finger swipes down.
        y_start = int(top + area_h * 0.15)
        y_end = min(int(bottom - area_h * 0.1), y_start + dist)
        swipe(device, x, y_start, x, y_end, duration_ms)
        expected = -(y_end - y_start)

    _log(f"[SCROLL] {direction} swipe y={y_start}->{y_end} expected_delta={expected}")
    return expected


def _scroll_and_capture(
    device,
    width: int,
    height: int,
    scroll_area: Tuple[int, int, int, int],
    direction: str,
    prev_nodes: List[Dict[str, Any]],
    distance_px: Optional[int] = None,
    duration_ms: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    expected = _scroll_once(
        device,
        width,
        height,
        scroll_area,
        direction,
        distance_px,
        duration_ms=duration_ms or 450,
    )
    time.sleep(0.5)
    xml = _dump_ui_xml(device)
    nodes = _parse_ui_nodes(xml)
    current_scroll_area = _find_scroll_area(nodes) or scroll_area
    actual = _compute_scroll_delta(prev_nodes, nodes, current_scroll_area)
    prev_sig = _screen_signature(prev_nodes, current_scroll_area)
    curr_sig = _screen_signature(nodes, current_scroll_area)
    screen_changed = prev_sig != curr_sig
    overlap = 0.0
    if prev_sig:
        overlap = len(prev_sig & curr_sig) / max(1, len(prev_sig))
    if actual is None:
        if screen_changed:
            actual = expected
            _log(f"[SCROLL] delta=unknown fallback={actual}")
        else:
            actual = 0
            _log("[SCROLL] no-move confirmed (signature unchanged)")
    else:
        # Guard against sign flips.
        if (direction == "down" and actual < 0) or (direction == "up" and actual > 0):
            if screen_changed:
                _log(f"[SCROLL] delta sign mismatch ({actual}); using fallback {expected}")
                actual = expected
            else:
                _log("[SCROLL] sign mismatch but signature unchanged; treating as no-move")
                actual = 0
        elif abs(actual) <= 5 and abs(expected) >= 200:
            if overlap >= 0.80:
                _log(f"[SCROLL] no-move confirmed (overlap={overlap:.2f})")
                actual = 0
            else:
                if screen_changed:
                    _log(f"[SCROLL] delta unreliable (0, overlap={overlap:.2f}); using fallback {expected}")
                    actual = expected
                else:
                    _log("[SCROLL] delta unreliable but signature unchanged; treating as no-move")
                    actual = 0
        elif abs(actual - expected) >= 400 and overlap < 0.60:
            if screen_changed:
                _log(f"[SCROLL] delta mismatch actual={actual} expected={expected} overlap={overlap:.2f}; using fallback")
                actual = expected
            else:
                _log("[SCROLL] delta mismatch but signature unchanged; treating as no-move")
                actual = 0
        else:
            _log(f"[SCROLL] delta=measured {actual}")
    return nodes, actual


def _scroll_to_top(
    device,
    width: int,
    height: int,
    max_attempts: int = 8,
) -> Tuple[List[Dict[str, Any]], Optional[Tuple[int, int, int, int]]]:
    """
    Best-effort reset to the top by scrolling up until no movement.
    Returns the last nodes and scroll_area.
    """
    xml = _dump_ui_xml(device)
    nodes = _parse_ui_nodes(xml)
    scroll_area = _find_scroll_area(nodes)
    if not scroll_area:
        return nodes, None
    no_move = 0
    attempts = 0
    while attempts < max_attempts and no_move < 2:
        prev_nodes = nodes
        nodes, delta = _scroll_and_capture(
            device, width, height, scroll_area, "up", prev_nodes
        )
        if abs(delta) <= 5:
            no_move += 1
        else:
            no_move = 0
        attempts += 1
    _log("[SCROLL] top reset complete")
    return nodes, scroll_area


def _build_ui_map(
    device,
    width: int,
    height: int,
    max_scrolls: int = 30,
) -> Dict[str, Any]:
    """
    One-pass scan from top to bottom building the UI map and biometrics.
    """
    ui_map = {
        "prompts": [],
        "photos": [],
        "poll": {"question": "", "options": []},
        "likes": [],
        "scroll_area": None,
        "scroll_history": [],
    }
    biometrics: Dict[str, Any] = {}

    # Start from the top for absolute mapping.
    nodes, scroll_area = _scroll_to_top(device, width, height)
    if not scroll_area:
        _log("[UI] No scrollable area found in XML.")
        return {"ui_map": ui_map, "biometrics": biometrics, "scroll_offset": 0, "nodes": nodes}

    ui_map["scroll_area"] = scroll_area
    scroll_offset = 0
    scrolls = 0
    no_move = 0
    did_hscroll = False
    last_counts: Optional[Tuple[int, int, int]] = None

    while True:
        # Extract biometrics visible on this screen.
        updates = _extract_biometrics_from_nodes(nodes, scroll_area)
        for k, v in updates.items():
            if k not in biometrics and v not in ("", None):
                biometrics[k] = v
                _log(f"[BIOMETRICS] {k} = {v}")
        # Attempt horizontal biometrics scroll once when the row is visible.
        if not did_hscroll and any(k in biometrics for k in ("Age", "Gender", "Sexuality")):
            nodes = _scan_biometrics_hscroll(device, nodes, scroll_area, biometrics)
            did_hscroll = True

        # Update UI map for prompts/polls/photos/likes.
        _update_ui_map(ui_map, nodes, scroll_area, scroll_offset)
        _log(
            f"[MAP] offset={scroll_offset} prompts={len(ui_map['prompts'])} "
            f"photos={len(ui_map['photos'])} poll_options={len(ui_map['poll']['options'])}"
        )
        current_counts = (
            len(ui_map["prompts"]),
            len(ui_map["photos"]),
            len(ui_map["poll"]["options"]),
        )
        if last_counts is None:
            last_counts = current_counts

        if scrolls >= max_scrolls:
            _log("[SCROLL] max_scrolls reached; stopping.")
            break

        prev_nodes = nodes
        nodes, delta = _scroll_and_capture(
            device, width, height, scroll_area, "down", prev_nodes
        )
        # Update scroll area if it changes.
        scroll_area = _find_scroll_area(nodes) or scroll_area
        ui_map["scroll_area"] = scroll_area

        scroll_offset += delta
        ui_map["scroll_history"].append(delta)
        scrolls += 1

        # If we didn't move and the map didn't grow, count towards bottom detection.
        if abs(delta) <= 5 and current_counts == last_counts:
            no_move += 1
        else:
            no_move = 0
            last_counts = current_counts

        if no_move >= 2:
            # Still process the last visible screen before exiting.
            _update_ui_map(ui_map, nodes, scroll_area, scroll_offset)
            _log("[SCROLL] No movement detected twice; likely bottom reached.")
            break

    _assign_like_buttons(ui_map)
    _dedupe_photos_by_like_bounds(ui_map)
    _assign_ids(ui_map)
    # Log final map summary for debugging.
    for p in ui_map.get("photos", []):
        _log(
            f"[MAP] photo id={p.get('id')} abs={p.get('abs_bounds')} like={p.get('like_bounds')}"
        )
    for p in ui_map.get("prompts", []):
        _log(
            f"[MAP] prompt id={p.get('id')} abs={p.get('abs_bounds')} like={p.get('like_bounds')}"
        )
    if ui_map.get("poll", {}).get("question"):
        _log(f"[MAP] poll question='{ui_map['poll'].get('question')}'")
        for opt in ui_map.get("poll", {}).get("options", []):
            _log(f"[MAP] poll option id={opt.get('id')} abs={opt.get('abs_bounds')}")
    return {
        "ui_map": ui_map,
        "biometrics": biometrics,
        "scroll_offset": scroll_offset,
        "nodes": nodes,
    }


def _scroll_to_offset(
    device,
    width: int,
    height: int,
    scroll_area: Tuple[int, int, int, int],
    current_offset: int,
    target_offset: int,
    max_steps: int = 20,
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Scroll until the desired absolute offset is reached.
    Returns (new_offset, last_nodes).
    """
    offset = current_offset
    area_h = max(1, scroll_area[3] - scroll_area[1])
    xml = _dump_ui_xml(device)
    nodes = _parse_ui_nodes(xml)
    steps = 0
    no_move = 0
    while abs(offset - target_offset) > 20 and steps < max_steps:
        delta = target_offset - offset
        direction = "down" if delta > 0 else "up"
        max_step = int(area_h * 0.55)
        step_px = min(abs(delta), max_step)
        if no_move >= 1:
            # If we didn't move, back off to a smaller nudge.
            step_px = min(step_px, int(area_h * 0.35))
        duration_ms = 380 if step_px >= int(area_h * 0.45) else 420
        prev_nodes = nodes
        nodes, actual = _scroll_and_capture(
            device,
            width,
            height,
            scroll_area,
            direction,
            prev_nodes,
            distance_px=step_px,
            duration_ms=duration_ms,
        )
        scroll_area = _find_scroll_area(nodes) or scroll_area
        offset += actual
        steps += 1
        _log(f"[SCROLL] offset now {offset} (target {target_offset})")
        if abs(actual) <= 5:
            no_move += 1
        else:
            no_move = 0
        if no_move >= 2:
            _log("[SCROLL] seek no-move twice; stopping early")
            break
    return offset, nodes


def _seek_target_on_screen(
    device,
    width: int,
    height: int,
    scroll_area: Tuple[int, int, int, int],
    current_offset: int,
    target_type: str,
    target_info: Dict[str, Any],
    desired_offset: int,
    max_steps: int = 12,
) -> Dict[str, Any]:
    """
    Scroll in small steps until the target is visible. This avoids overshooting
    when offset deltas are unreliable.
    Returns a dict with updated nodes/scroll_area/offset and any found bounds.
    """
    offset = current_offset
    xml = _dump_ui_xml(device)
    nodes = _parse_ui_nodes(xml)
    scroll_area = _find_scroll_area(nodes) or scroll_area
    area_h = max(1, scroll_area[3] - scroll_area[1])
    no_move = 0
    steps = 0

    while steps < max_steps:
        found: Dict[str, Any] = {}
        if target_type == "prompt":
            prompt_bounds = _find_prompt_bounds_by_text(
                nodes,
                target_info.get("prompt", ""),
                target_info.get("answer", ""),
            )
            if prompt_bounds:
                _log(f"[SEEK] prompt visible at {prompt_bounds}")
                found["prompt_bounds"] = prompt_bounds
        elif target_type == "poll":
            option_bounds = _find_poll_option_bounds_by_text(
                nodes, target_info.get("option_text", "")
            )
            if option_bounds:
                _log(f"[SEEK] poll option visible at {option_bounds}")
                found["poll_bounds"] = option_bounds
        elif target_type == "photo":
            target_hash = target_info.get("photo_hash")
            target_photo_bounds = target_info.get("photo_bounds")
            target_abs_center_y = None
            expected_screen_y = None
            if target_photo_bounds:
                target_abs_center_y = int(
                    (target_photo_bounds[1] + target_photo_bounds[3]) / 2
                )
                expected_screen_y = int(target_abs_center_y - offset)
            if target_hash is not None:
                match_bounds, dist = _match_photo_bounds_by_hash(
                    device,
                    width,
                    height,
                    nodes,
                    scroll_area,
                    int(target_hash),
                    expected_screen_y=expected_screen_y,
                    max_dist=18,
                    square_only=True,
                )
                if match_bounds:
                    _log(f"[SEEK] photo hash visible at {match_bounds} dist={dist}")
                    found["photo_match_bounds"] = match_bounds
            if not found and expected_screen_y is not None:
                photo_bounds = _find_visible_photo_bounds(
                    nodes, scroll_area, expected_screen_y
                )
                if photo_bounds and _is_square_bounds(photo_bounds):
                    _log(f"[SEEK] photo candidate near expected y at {photo_bounds}")
                    found["photo_bounds"] = photo_bounds

        if found:
            return {
                "nodes": nodes,
                "scroll_area": scroll_area,
                "scroll_offset": offset,
                **found,
            }

        # Not visible yet; move toward desired offset in smaller steps.
        delta = desired_offset - offset
        if abs(delta) <= 20:
            step_px = 140
        else:
            step_px = min(abs(delta), int(area_h * 0.45))
        direction = "down" if delta > 0 else "up"
        prev_nodes = nodes
        nodes, actual = _scroll_and_capture(
            device,
            width,
            height,
            scroll_area,
            direction,
            prev_nodes,
            distance_px=step_px,
            duration_ms=420,
        )
        scroll_area = _find_scroll_area(nodes) or scroll_area
        offset += actual
        _log(f"[SEEK] step offset now {offset} (target {desired_offset})")
        steps += 1
        if abs(actual) <= 5:
            no_move += 1
        else:
            no_move = 0
        if no_move >= 2:
            _log("[SEEK] no-move twice; stopping")
            break

    return {
        "nodes": nodes,
        "scroll_area": scroll_area,
        "scroll_offset": offset,
    }


def _seek_photo_by_hash(
    device,
    width: int,
    height: int,
    scroll_area: Tuple[int, int, int, int],
    target_hash: int,
    max_steps: int = 20,
    max_dist: int = 18,
) -> Dict[str, Any]:
    """
    Photo-only re-acquire: scroll from top in fixed steps and hash-match
    the visible square photo. Ignores offsets to avoid drift.
    """
    nodes, scroll_area = _scroll_to_top(device, width, height)
    if not scroll_area:
        return {"nodes": nodes, "scroll_area": scroll_area}
    offset = 0
    area_h = max(1, scroll_area[3] - scroll_area[1])
    step_px = int(area_h * 0.45)
    no_move = 0
    steps = 0

    while steps < max_steps:
        photo_bounds = _find_primary_photo_bounds(nodes, scroll_area)
        if photo_bounds:
            nodes, offset, photo_bounds = _ensure_photo_square(
                device,
                width,
                height,
                scroll_area,
                nodes,
                offset,
                photo_bounds,
            )
            scroll_area = _find_scroll_area(nodes) or scroll_area
            if photo_bounds and _is_square_bounds(photo_bounds):
                h = _compute_center_ahash_from_bounds(
                    device, photo_bounds, width, height
                )
                dist = _ahash_distance(h, target_hash) if h is not None else None
                _log(f"[SEEK-PHOTO] candidate bounds={photo_bounds} dist={dist}")
                if dist is not None and dist <= max_dist:
                    like_bounds, like_desc = _find_like_button_in_photo(
                        nodes, photo_bounds
                    )
                    if like_bounds:
                        _log(
                            f"[SEEK-PHOTO] match dist={dist} like_bounds={like_bounds}"
                        )
                        return {
                            "nodes": nodes,
                            "scroll_area": scroll_area,
                            "scroll_offset": offset,
                            "tap_bounds": like_bounds,
                            "tap_desc": like_desc,
                        }

        prev_nodes = nodes
        nodes, delta = _scroll_and_capture(
            device,
            width,
            height,
            scroll_area,
            "down",
            prev_nodes,
            distance_px=step_px,
            duration_ms=420,
        )
        scroll_area = _find_scroll_area(nodes) or scroll_area
        offset += delta
        _log(f"[SEEK-PHOTO] step {steps+1} offset={offset}")
        steps += 1
        if abs(delta) <= 5:
            no_move += 1
        else:
            no_move = 0
        if no_move >= 2:
            _log("[SEEK-PHOTO] no-move twice; stopping")
            break

    return {"nodes": nodes, "scroll_area": scroll_area, "scroll_offset": offset}


def _seek_photo_by_index(
    device,
    width: int,
    height: int,
    scroll_area: Tuple[int, int, int, int],
    target_index: int,
    target_hash: Optional[int] = None,
    max_steps: int = 25,
    max_dist: int = 18,
) -> Dict[str, Any]:
    """
    Photo-only re-acquire by index: scroll from top and count square photos.
    Uses center-hash to avoid double-counting the same photo.
    """
    nodes, scroll_area = _scroll_to_top(device, width, height)
    if not scroll_area:
        return {"nodes": nodes, "scroll_area": scroll_area}
    offset = 0
    area_h = max(1, scroll_area[3] - scroll_area[1])
    step_px = int(area_h * 0.45)
    no_move = 0
    steps = 0
    count = 0
    last_hash: Optional[int] = None

    while steps < max_steps:
        photo_bounds = _find_primary_photo_bounds(nodes, scroll_area)
        if photo_bounds:
            nodes, offset, photo_bounds = _ensure_photo_square(
                device,
                width,
                height,
                scroll_area,
                nodes,
                offset,
                photo_bounds,
            )
            scroll_area = _find_scroll_area(nodes) or scroll_area
            if photo_bounds and _is_square_bounds(photo_bounds):
                h = _compute_center_ahash_from_bounds(
                    device, photo_bounds, width, height
                )
                dist = _ahash_distance(h, target_hash) if (h is not None and target_hash is not None) else None
                _log(f"[SEEK-PHOTO] candidate bounds={photo_bounds} dist={dist}")
                is_new = True
                if h is not None and last_hash is not None:
                    if _ahash_distance(h, last_hash) <= 6:
                        is_new = False
                if is_new:
                    count += 1
                    last_hash = h
                    _log(f"[SEEK-PHOTO] count={count} target={target_index}")
                    if count == target_index:
                        like_bounds, like_desc = _find_like_button_in_photo(
                            nodes, photo_bounds
                        )
                        if like_bounds:
                            if dist is not None and dist > max_dist:
                                _log(f"[SEEK-PHOTO] warning: hash dist {dist} > {max_dist} at target index")
                            _log(
                                f"[SEEK-PHOTO] index match like_bounds={like_bounds}"
                            )
                            return {
                                "nodes": nodes,
                                "scroll_area": scroll_area,
                                "scroll_offset": offset,
                                "tap_bounds": like_bounds,
                                "tap_desc": like_desc,
                            }

        prev_nodes = nodes
        nodes, delta = _scroll_and_capture(
            device,
            width,
            height,
            scroll_area,
            "down",
            prev_nodes,
            distance_px=step_px,
            duration_ms=420,
        )
        scroll_area = _find_scroll_area(nodes) or scroll_area
        offset += delta
        _log(f"[SEEK-PHOTO] step {steps+1} offset={offset}")
        steps += 1
        if abs(delta) <= 5:
            no_move += 1
        else:
            no_move = 0
        if no_move >= 2:
            _log("[SEEK-PHOTO] no-move twice; stopping")
            break

    return {"nodes": nodes, "scroll_area": scroll_area, "scroll_offset": offset}


def _compute_desired_offset(
    abs_bounds: Tuple[int, int, int, int],
    scroll_area: Tuple[int, int, int, int],
) -> int:
    """
    Compute a scroll_offset that keeps the target bounds fully visible.
    """
    top, bottom = scroll_area[1], scroll_area[3]
    abs_top, abs_bottom = abs_bounds[1], abs_bounds[3]
    scroll_center = int((top + bottom) / 2)
    target_center = int((abs_top + abs_bottom) / 2)
    desired = target_center - scroll_center
    # Adjust to ensure fully visible.
    if abs_top - desired < top:
        desired = abs_top - top
    if abs_bottom - desired > bottom:
        desired = abs_bottom - bottom
    return max(0, int(desired))


def _capture_crop_from_device(
    device,
    bounds: Tuple[int, int, int, int],
    out_name: str,
    width: int,
    height: int,
) -> str:
    """
    Capture a full screencap and crop to bounds.
    """
    x1, y1, x2, y2 = bounds
    x1 = max(0, min(width - 1, x1))
    x2 = max(1, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(1, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid crop bounds")

    img_bytes = device.screencap()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    crop = img.crop((x1, y1, x2, y2))

    os.makedirs(os.path.join("images", "crops"), exist_ok=True)
    ts = int(time.time() * 1000)
    out_path = os.path.join("images", "crops", f"{ts}_{out_name}.png")
    crop.save(out_path)
    return out_path


def _clear_crops_folder() -> None:
    crops_dir = os.path.join("images", "crops")
    if not os.path.isdir(crops_dir):
        return
    removed = 0
    for name in os.listdir(crops_dir):
        if not name.lower().endswith(".png"):
            continue
        path = os.path.join(crops_dir, name)
        try:
            os.remove(path)
            removed += 1
        except Exception as e:
            _log(f"[PHOTO] failed to remove crop {path}: {e}")
    if removed:
        _log(f"[PHOTO] cleared {removed} crop files")


def _find_visible_photo_bounds(
    nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
    expected_screen_y: int,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Find the visible photo ImageView bounds closest to the expected Y on screen.
    """
    top, bottom = scroll_area[1], scroll_area[3]
    best = None
    best_dist = None
    for n in nodes:
        if n.get("cls") != "android.widget.ImageView":
            continue
        cd = (n.get("content_desc") or "").lower()
        if "photo" not in cd:
            continue
        b = n.get("bounds")
        if not b:
            continue
        if b[1] >= bottom or b[3] <= top:
            continue
        cy = _bounds_center(b)[1]
        dist = abs(cy - expected_screen_y)
        if best_dist is None or dist < best_dist:
            best = b
            best_dist = dist
    return best


def _capture_photo_crops(
    device,
    width: int,
    height: int,
    ui_map: Dict[str, Any],
    start_offset: int,
) -> Tuple[List[str], int]:
    """
    Scroll to each photo and capture exact cropped image.
    """
    scroll_area = ui_map.get("scroll_area")
    if not scroll_area:
        _log("[PHOTO] Missing scroll_area; cannot capture crops.")
        return [], start_offset

    photo_paths: List[str] = []
    offset = start_offset
    # Select unique photos by like_bounds (fallback to coarse abs_center_y) without mutating the map.
    photos_to_capture: List[Dict[str, Any]] = []
    seen_like_bounds = set()
    for photo in ui_map.get("photos", []):
        lb = photo.get("like_bounds")
        if lb:
            key = ("like",) + tuple(lb)
        else:
            cy = photo.get("abs_center_y") or 0
            key = ("y", int(round(cy / 50.0)) * 50)
        if key in seen_like_bounds:
            _log(f"[PHOTO] skipping duplicate photo id={photo.get('id')} like_bounds={key}")
            continue
        seen_like_bounds.add(key)
        photos_to_capture.append(photo)
    if len(photos_to_capture) > 6:
        photos_to_capture = photos_to_capture[:6]

    for photo in photos_to_capture:
        abs_bounds = photo.get("abs_bounds")
        if not abs_bounds:
            continue
        desired_offset = _compute_desired_offset(abs_bounds, scroll_area)
        offset, _ = _scroll_to_offset(
            device, width, height, scroll_area, offset, desired_offset
        )
        # Try to re-read the current screen bounds for this photo.
        xml = _dump_ui_xml(device)
        nodes = _parse_ui_nodes(xml)
        expected_screen_y = int((abs_bounds[1] + abs_bounds[3]) / 2 - offset)
        visible_bounds = _find_visible_photo_bounds(nodes, scroll_area, expected_screen_y)
        if visible_bounds:
            viewport_bounds = visible_bounds
        else:
            viewport_bounds = (
                abs_bounds[0],
                abs_bounds[1] - offset,
                abs_bounds[2],
                abs_bounds[3] - offset,
            )

        vb_w = viewport_bounds[2] - viewport_bounds[0]
        vb_h = viewport_bounds[3] - viewport_bounds[1]
        is_square = abs(vb_w - vb_h) <= 12
        _log(
            f"[PHOTO] {photo.get('id')} bounds={viewport_bounds} size={vb_w}x{vb_h} square={'yes' if is_square else 'no'}"
        )

        # Micro-scroll to try to get a full square photo visible.
        attempts = 0
        while not is_square and attempts < 4:
            top_edge = scroll_area[1]
            bottom_edge = scroll_area[3]
            direction = None
            if viewport_bounds[1] <= top_edge + 3:
                direction = "up"
            elif viewport_bounds[3] >= bottom_edge - 3:
                direction = "down"
            if not direction:
                break
            prev_nodes = nodes
            nodes, delta = _scroll_and_capture(
                device,
                width,
                height,
                scroll_area,
                direction,
                prev_nodes,
                distance_px=140,
            )
            offset += delta
            expected_screen_y = int((abs_bounds[1] + abs_bounds[3]) / 2 - offset)
            visible_bounds = _find_visible_photo_bounds(nodes, scroll_area, expected_screen_y)
            if visible_bounds:
                viewport_bounds = visible_bounds
                vb_w = viewport_bounds[2] - viewport_bounds[0]
                vb_h = viewport_bounds[3] - viewport_bounds[1]
                is_square = abs(vb_w - vb_h) <= 12
                _log(
                    f"[PHOTO] micro-scroll {attempts+1} dir={direction} size={vb_w}x{vb_h} square={'yes' if is_square else 'no'}"
                )
            attempts += 1
        try:
            crop_path = _capture_crop_from_device(
                device, viewport_bounds, photo.get("id", "photo"), width, height
            )
            photo["crop_path"] = crop_path
            photo_paths.append(crop_path)
            _log(f"[PHOTO] captured {photo.get('id')} -> {crop_path}")
        except Exception as e:
            _log(f"[PHOTO] capture failed for {photo.get('id')}: {e}")
    if len(photo_paths) < 6:
        _log(f"[PHOTO] Warning: captured {len(photo_paths)} unique photos (expected 6)")
    return photo_paths, offset


def _scan_profile_single_pass(
    device,
    width: int,
    height: int,
    max_scrolls: int = 40,
    scroll_step_px: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Single-pass slow scan: extract text + biometrics, capture photos as they appear.
    """
    ui_map = {
        "prompts": [],
        "photos": [],
        "poll": {"question": "", "options": []},
        "likes": [],
        "scroll_area": None,
        "scroll_history": [],
    }
    biometrics: Dict[str, Any] = {}
    photo_paths: List[str] = []
    seen_photo_keys: Set[int] = set()
    last_capture_abs_top: Optional[int] = None
    last_capture_height: Optional[int] = None
    skip_photo_capture_once = False

    xml = _dump_ui_xml(device)
    nodes = _parse_ui_nodes(xml)
    scroll_area = _find_scroll_area(nodes)
    if not scroll_area:
        _log("[UI] No scrollable area found in XML.")
        return {"ui_map": ui_map, "biometrics": biometrics, "photo_paths": photo_paths, "scroll_offset": 0}
    ui_map["scroll_area"] = scroll_area
    offset = 0
    did_hscroll = False
    no_move = 0
    scrolls = 0

    while True:
        # Extract biometrics visible on this screen.
        updates = _extract_biometrics_from_nodes(nodes, scroll_area)
        for k, v in updates.items():
            if k not in biometrics and v not in ("", None):
                biometrics[k] = v
                _log(f"[BIOMETRICS] {k} = {v}")

        # Horizontal biometrics scroll (once), stop when no new values appear.
        if not did_hscroll and any(k in biometrics for k in ("Age", "Gender", "Sexuality")):
            nodes = _scan_biometrics_hscroll(device, nodes, scroll_area, biometrics)
            did_hscroll = True
            skip_photo_capture_once = True

        # Update prompts/polls/likes.
        _update_ui_map_text_only(ui_map, nodes, scroll_area, offset)

        # Capture primary photo if present and new.
        photo_bounds = _find_primary_photo_bounds(nodes, scroll_area)
        if photo_bounds:
            if skip_photo_capture_once:
                _log("[PHOTO] skip capture immediately after hscroll iteration")
                skip_photo_capture_once = False
            else:
                # Micro-scroll if needed to get a full square photo.
                nodes, offset, photo_bounds = _ensure_photo_square(
                    device, width, height, scroll_area, nodes, offset, photo_bounds
                )
                scroll_area = _find_scroll_area(nodes) or scroll_area
                ui_map["scroll_area"] = scroll_area
                if photo_bounds:
                    vb_w = photo_bounds[2] - photo_bounds[0]
                    vb_h = photo_bounds[3] - photo_bounds[1]
                    is_square = abs(vb_w - vb_h) <= 12
                    if not is_square:
                        _log(f"[PHOTO] skip non-square size={vb_w}x{vb_h}")
                    else:
                        abs_bounds = (
                            photo_bounds[0],
                            photo_bounds[1] + offset,
                            photo_bounds[2],
                            photo_bounds[3] + offset,
                        )
                        abs_top = abs_bounds[1]
                        key = int(round(abs_top / 50.0)) * 50
                        min_gap = int((last_capture_height or vb_h) * 0.6)
                        if last_capture_abs_top is not None:
                            gap = abs_top - last_capture_abs_top
                            if gap <= 0 or gap < min_gap:
                                _log(
                                    f"[PHOTO] skip candidate abs_top={abs_top} gap={gap} min_gap={min_gap}"
                                )
                                photo_bounds = None
                        if photo_bounds:
                            if key in seen_photo_keys:
                                _log(f"[PHOTO] skip duplicate abs_top={abs_top} key={key}")
                            else:
                                like_bounds, like_desc = _find_like_button_in_photo(nodes, photo_bounds)
                                like_abs = None
                                if like_bounds:
                                    like_abs = (
                                        like_bounds[0],
                                        like_bounds[1] + offset,
                                        like_bounds[2],
                                        like_bounds[3] + offset,
                                    )

                                try:
                                    crop_path = _capture_crop_from_device(
                                        device,
                                        photo_bounds,
                                        f"photo_{len(ui_map['photos'])+1}",
                                        width,
                                        height,
                                    )
                                    photo_paths.append(crop_path)
                                except Exception as e:
                                    _log(f"[PHOTO] capture failed: {e}")
                                    crop_path = ""

                                photo_hash = (
                                    _compute_center_ahash_from_file(crop_path) if crop_path else None
                                )
                                like_center = _bounds_center(like_abs) if like_abs else None
                                ui_map["photos"].append(
                                    {
                                        "content_desc": "photo",
                                        "abs_bounds": abs_bounds,
                                        "abs_center_y": int((abs_bounds[1] + abs_bounds[3]) / 2),
                                        "like_bounds": like_abs,
                                        "like_desc": like_desc,
                                        "crop_path": crop_path,
                                        "hash": photo_hash,
                                        "abs_top": abs_top,
                                        "like_center": like_center,
                                    }
                                )
                                seen_photo_keys.add(key)
                                last_capture_abs_top = abs_top
                                last_capture_height = vb_h
                                _log(
                                    f"[PHOTO] captured abs_top={abs_top} abs_bounds={abs_bounds} "
                                    f"like_abs={like_abs} like_center={like_center}"
                                )
        elif skip_photo_capture_once:
            # Ensure we only skip once even if no photo was visible.
            skip_photo_capture_once = False

        # Scroll down for next screen.
        if scrolls >= max_scrolls:
            _log("[SCROLL] max_scrolls reached; stopping.")
            break
        prev_nodes = nodes
        nodes, delta = _scroll_and_capture(
            device,
            width,
            height,
            scroll_area,
            "down",
            prev_nodes,
            distance_px=scroll_step_px,
        )
        scroll_area = _find_scroll_area(nodes) or scroll_area
        ui_map["scroll_area"] = scroll_area
        offset += delta
        ui_map["scroll_history"].append(delta)
        scrolls += 1

        if abs(delta) <= 5:
            no_move += 1
        else:
            no_move = 0
        if no_move >= 2:
            _log("[SCROLL] No movement detected twice; likely bottom reached.")
            break

    _assign_like_buttons(ui_map)
    _assign_ids(ui_map)
    _log(
        f"[UI] scan done prompts={len(ui_map.get('prompts', []))} "
        f"photos={len(ui_map.get('photos', []))} poll_options={len(ui_map.get('poll', {}).get('options', []))}"
    )
    for p in ui_map.get("photos", []):
        _log(
            f"[MAP] photo id={p.get('id')} abs_top={p.get('abs_top')} like_abs={p.get('like_bounds')}"
        )
    return {
        "ui_map": ui_map,
        "biometrics": biometrics,
        "photo_paths": photo_paths,
        "scroll_offset": offset,
        "scroll_area": scroll_area,
        "nodes": nodes,
    }


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
        "Apparent Upper Body Proportions",
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


def _resolve_target_from_ui_map(
    ui_map: Dict[str, Any],
    target_id: str,
) -> Dict[str, Any]:
    target_id = (target_id or "").strip()
    if not target_id:
        return {}

    if target_id.startswith("prompt_"):
        for p in ui_map.get("prompts", []):
            if p.get("id") == target_id:
                if not p.get("like_bounds"):
                    return {
                        "type": "prompt",
                        "abs_bounds": None,
                        "error": "missing_like_bounds",
                        "prompt": p.get("prompt", ""),
                        "answer": p.get("answer", ""),
                        "prompt_bounds": p.get("abs_bounds"),
                    }
                return {
                    "type": "prompt",
                    "abs_bounds": p.get("like_bounds"),
                    "prompt": p.get("prompt", ""),
                    "answer": p.get("answer", ""),
                    "prompt_bounds": p.get("abs_bounds"),
                }
    if target_id.startswith("photo_"):
        for p in ui_map.get("photos", []):
            if p.get("id") == target_id:
                if not p.get("like_bounds"):
                    return {
                        "type": "photo",
                        "abs_bounds": None,
                        "error": "missing_like_bounds",
                        "photo_bounds": p.get("abs_bounds"),
                        "photo_hash": p.get("hash"),
                    }
                return {
                    "type": "photo",
                    "abs_bounds": p.get("like_bounds"),
                    "photo_bounds": p.get("abs_bounds"),
                    "photo_hash": p.get("hash"),
                }
    if target_id.startswith("poll_"):
        for opt in ui_map.get("poll", {}).get("options", []):
            if opt.get("id") == target_id:
                return {
                    "type": "poll",
                    "abs_bounds": opt.get("abs_bounds"),
                    "option_text": opt.get("text", ""),
                }
    return {}


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
        record("Core Biometrics", "Dating Intentions", dating, -20)

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


def _classify_preference_flag(
    long_score: int,
    short_score: int,
    t_long: int = 15,
    t_short: int = 20,
    dominance_margin: int = 10,
) -> str:
    long_excess = long_score - t_long
    short_excess = short_score - t_short
    if long_score >= t_long and long_excess >= short_excess + dominance_margin:
        return "LONG"
    if short_score >= t_short and short_excess >= long_excess + dominance_margin:
        return "SHORT"
    return "NONE"


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

    device = s.get("device")
    width = int(s.get("width") or 0)
    height = int(s.get("height") or 0)
    if not device or not width or not height:
        print("Device/size missing; cannot proceed.")
        return 1

    _clear_crops_folder()
    _log("[UI] Single-pass scan (slow scroll, capture as you go)...")
    scan_result = _scan_profile_single_pass(device, width, height, max_scrolls=40, scroll_step_px=700)
    ui_map = scan_result.get("ui_map", {})
    biometrics = scan_result.get("biometrics", {})
    photo_paths = scan_result.get("photo_paths", [])
    scroll_offset = int(scan_result.get("scroll_offset", 0))
    scroll_area = scan_result.get("scroll_area")

    _log(f"[LLM1] Sending {len(photo_paths)} photos for visual analysis")
    llm1_result, llm1_meta = run_llm1_visual(
        photo_paths,
        model=os.getenv("LLM_SMALL_MODEL") or os.getenv("GEMINI_SMALL_MODEL") or None,
    )
    extracted = _build_extracted_profile(biometrics, ui_map, llm1_result)

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
            target_info = _resolve_target_from_ui_map(ui_map, target_id)
            target_action = {"target_id": target_id, **target_info}
            target_type = target_info.get("type", "")
            if target_type == "photo":
                target_hash = target_info.get("photo_hash")
                target_index = None
                try:
                    target_index = int(str(target_id).split("_", 1)[1])
                except Exception:
                    target_index = None
                if target_hash is None or not scroll_area:
                    print("[TARGET] missing photo hash or scroll area; skipping tap")
                elif not target_index:
                    print("[TARGET] missing photo index; skipping tap")
                else:
                    seek_photo = _seek_photo_by_index(
                        device,
                        width,
                        height,
                        scroll_area,
                        int(target_index),
                        target_hash=int(target_hash),
                    )
                    cur_nodes = seek_photo.get("nodes")
                    cur_scroll_area = seek_photo.get("scroll_area") or scroll_area
                    tap_bounds = seek_photo.get("tap_bounds")
                    tap_desc = seek_photo.get("tap_desc", "Like photo")
                    if tap_bounds:
                        tap_x, tap_y = _bounds_center(tap_bounds)
                        tap_x = max(0, min(width - 1, tap_x))
                        tap_y = max(0, min(height - 1, tap_y))
                        print(
                            f"[TARGET] photo tap bounds={tap_bounds} desc='{tap_desc}'"
                        )
                        try:
                            tap(device, tap_x, tap_y)
                            print(f"[TARGET] tap issued at ({tap_x}, {tap_y})")
                        except Exception as e:
                            print(f"[TARGET] tap failed: {e}")
                        target_action["tap_coords"] = [tap_x, tap_y]
                        target_action["tap_like"] = True
                        # Best-effort post-tap confirmation.
                        time.sleep(0.35)
                        post_xml = _dump_ui_xml(device)
                        post_nodes = _parse_ui_nodes(post_xml)
                        post_bounds, _ = _find_like_button_near_expected(
                            post_nodes, cur_scroll_area, "photo", tap_y
                        )
                        if _bounds_close(post_bounds, tap_bounds):
                            print("[TARGET] like button still present near tap (not confirmed)")
                        else:
                            print("[TARGET] like button not found near tap (likely tapped)")
                    else:
                        print("[TARGET] photo not found on-screen; skipping tap")
            elif target_info.get("abs_bounds") and scroll_area:
                target_bounds = target_info["abs_bounds"]
                focus_bounds = target_bounds
                if target_type == "photo" and target_info.get("photo_bounds"):
                    focus_bounds = target_info.get("photo_bounds")
                if target_type == "prompt" and target_info.get("prompt_bounds"):
                    focus_bounds = target_info.get("prompt_bounds")
                desired_offset = _compute_desired_offset(focus_bounds, scroll_area)
                seek = _seek_target_on_screen(
                    device,
                    width,
                    height,
                    scroll_area,
                    scroll_offset,
                    target_type,
                    target_info,
                    desired_offset,
                )
                scroll_offset = seek.get("scroll_offset", scroll_offset)
                cur_nodes = seek.get("nodes") or _parse_ui_nodes(_dump_ui_xml(device))
                cur_scroll_area = seek.get("scroll_area") or _find_scroll_area(cur_nodes) or scroll_area
                expected_screen_y = int(
                    (target_bounds[1] + target_bounds[3]) / 2 - scroll_offset
                )
                tap_bounds = None
                tap_desc = ""
                if target_type == "prompt":
                    prompt_bounds = seek.get("prompt_bounds")
                    if not prompt_bounds:
                        prompt_bounds = _find_prompt_bounds_by_text(
                            cur_nodes,
                            target_info.get("prompt", ""),
                            target_info.get("answer", ""),
                        )
                    if prompt_bounds:
                        tap_bounds, tap_desc = _find_like_button_near_bounds_screen(
                            cur_nodes, prompt_bounds, "prompt"
                        )
                        print(f"[TARGET] prompt found on-screen at {prompt_bounds}")
                    else:
                        print("[TARGET] prompt not found on-screen; falling back to expected Y")
                        tap_bounds, tap_desc = _find_like_button_near_expected(
                            cur_nodes, cur_scroll_area, "prompt", expected_screen_y
                        )
                elif target_type == "poll":
                    option_bounds = seek.get("poll_bounds")
                    if not option_bounds:
                        option_bounds = _find_poll_option_bounds_by_text(
                            cur_nodes, target_info.get("option_text", "")
                        )
                    if option_bounds:
                        tap_bounds = option_bounds
                        tap_desc = "poll_option"
                        print(f"[TARGET] poll option found on-screen at {option_bounds}")
                    else:
                        print("[TARGET] poll option not found on-screen; skipping tap")
                elif target_type == "photo":
                    target_hash = target_info.get("photo_hash")
                    target_photo_bounds = target_info.get("photo_bounds")
                    target_abs_center_y = None
                    if target_photo_bounds:
                        target_abs_center_y = int(
                            (target_photo_bounds[1] + target_photo_bounds[3]) / 2
                        )
                        expected_screen_y = int(target_abs_center_y - scroll_offset)

                    photo_bounds = seek.get("photo_bounds")
                    if not photo_bounds and target_abs_center_y is not None:
                        photo_bounds = _find_visible_photo_bounds(
                            cur_nodes, cur_scroll_area, expected_screen_y
                        )
                    if photo_bounds:
                        cur_nodes, scroll_offset, photo_bounds = _ensure_photo_square(
                            device,
                            width,
                            height,
                            cur_scroll_area,
                            cur_nodes,
                            scroll_offset,
                            photo_bounds,
                            target_abs_center_y=target_abs_center_y,
                        )
                        cur_scroll_area = _find_scroll_area(cur_nodes) or cur_scroll_area
                        if target_abs_center_y is not None:
                            expected_screen_y = int(target_abs_center_y - scroll_offset)

                    if target_hash is None:
                        print("[TARGET] missing photo hash; skipping hash match")
                    match_bounds = seek.get("photo_match_bounds")
                    dist = None
                    if target_hash is not None and not match_bounds:
                        match_bounds, dist = _match_photo_bounds_by_hash(
                            device,
                            width,
                            height,
                            cur_nodes,
                            cur_scroll_area,
                            int(target_hash),
                            expected_screen_y=expected_screen_y,
                            max_dist=18,
                            square_only=True,
                        )
                        if match_bounds:
                            tap_bounds, tap_desc = _find_like_button_in_photo(
                                cur_nodes, match_bounds
                            )
                            print(
                                f"[TARGET] photo hash matched bounds={match_bounds} dist={dist}"
                            )

                    if not tap_bounds and photo_bounds:
                        dy = None
                        if expected_screen_y is not None:
                            dy = abs(_bounds_center(photo_bounds)[1] - expected_screen_y)
                        if _is_square_bounds(photo_bounds) and (dy is None or dy <= 220):
                            tap_bounds, tap_desc = _find_like_button_in_photo(
                                cur_nodes, photo_bounds
                            )
                            print(
                                f"[TARGET] using closest square photo by y dist={dy}"
                            )
                    if not tap_bounds:
                        print("[TARGET] photo not found on-screen; skipping tap")
                else:
                    tap_bounds, tap_desc = _find_like_button_near_expected(
                        cur_nodes, cur_scroll_area, target_type, expected_screen_y
                    )
                if not tap_bounds:
                    if target_type in {"photo", "poll"}:
                        print(f"[TARGET] no bounds resolved for {target_type}; skipping tap")
                    else:
                        # Fallback: translate abs bounds to screen bounds using current offset.
                        tap_bounds = (
                            target_bounds[0],
                            target_bounds[1] - scroll_offset,
                            target_bounds[2],
                            target_bounds[3] - scroll_offset,
                        )
                if tap_bounds:
                    tap_x, tap_y = _bounds_center(tap_bounds)
                    tap_x = max(0, min(width - 1, tap_x))
                    tap_y = max(0, min(height - 1, tap_y))
                    print(
                        f"[TARGET] tap bounds={tap_bounds} desc='{tap_desc}' expected_y={expected_screen_y}"
                    )
                    try:
                        tap(device, tap_x, tap_y)
                        print(f"[TARGET] tap issued at ({tap_x}, {tap_y})")
                    except Exception as e:
                        print(f"[TARGET] tap failed: {e}")
                    target_action["tap_coords"] = [tap_x, tap_y]
                    target_action["tap_like"] = True
                    # Best-effort post-tap confirmation (skip for polls).
                    if target_type != "poll":
                        time.sleep(0.35)
                        post_xml = _dump_ui_xml(device)
                        post_nodes = _parse_ui_nodes(post_xml)
                        post_bounds, _ = _find_like_button_near_expected(
                            post_nodes, cur_scroll_area, target_type, tap_y
                        )
                        if _bounds_close(post_bounds, tap_bounds):
                            print("[TARGET] like button still present near tap (not confirmed)")
                        else:
                            print("[TARGET] like button not found near tap (likely tapped)")
                else:
                    print("[TARGET] no tap bounds resolved; skipping tap")
            else:
                print("[TARGET] missing bounds; skipping tap")

    out = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "llm_provider": os.getenv("LLM_PROVIDER", ""),
            "model": os.getenv("LLM_SMALL_MODEL") or os.getenv("GEMINI_SMALL_MODEL") or "",
            "images_count": llm1_meta.get("images_count"),
            "images_paths": llm1_meta.get("images_paths", []) or photo_paths,
            "timings": {},
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
        "ui_map_summary": {
            "prompts": len(ui_map.get("prompts", [])),
            "photos": len(ui_map.get("photos", [])),
            "poll_options": len(ui_map.get("poll", {}).get("options", [])),
        },
        "profile_eval": eval_result,
        "long_score_result": long_score_result,
        "short_score_result": short_score_result,
        "score_table_long": score_table_long,
        "score_table_short": score_table_short,
        "score_table": score_table,
    }

    if _is_run_json_enabled():
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

    long_score = long_score_result.get("score", 0) if isinstance(long_score_result, dict) else 0
    short_score = short_score_result.get("score", 0) if isinstance(short_score_result, dict) else 0
    preference_flag = _classify_preference_flag(long_score, short_score)
    print("\n=== Preference Flag ===")
    print(
        f"classification={preference_flag} "
        f"(long_score={long_score}, short_score={short_score}, "
        "t_long=15, t_short=20, dominance_margin=10)"
    )

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
