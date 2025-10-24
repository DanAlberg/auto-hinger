# app/prompt_engine.py

COMEDIC_KEY = "comedic"
FLIRTY_KEY = "flirty"
STRAIGHTFORWARD_KEY = "straightforward"

COMEDIC_TEMPLATE = (
    "The profile mentions '{keyword}'. That's hilarious! "
    "Please create a short, witty comment referencing that."
)
FLIRTY_TEMPLATE = (
    "This person loves '{keyword}'. Write a playful invitation "
    "asking them about it in a flirty, friendly way."
)
STRAIGHTFORWARD_TEMPLATE = (
    "They mentioned '{keyword}'. Generate a direct, polite invitation "
    "to discuss that topic over coffee."
)

# Global weights for each style
TEMPLATE_WEIGHTS = {COMEDIC_KEY: 1.0, FLIRTY_KEY: 1.0, STRAIGHTFORWARD_KEY: 1.0}


def update_template_weights(success_rates: dict):
    """
    If comedic style yields a higher success rate, automatically adjust
    to favor comedic, etc.
    """
    if not success_rates:
        return

    best_template = max(success_rates, key=success_rates.get)
    # Reset all weights to a baseline
    baseline = 1.0
    for key in TEMPLATE_WEIGHTS:
        TEMPLATE_WEIGHTS[key] = baseline

    # Increase the weight of whichever template is best
    # (for example, we identify comedic by checking if "hilarious" is in the template string)
    if "hilarious" in best_template:
        TEMPLATE_WEIGHTS[COMEDIC_KEY] = baseline + 0.5
    elif "flirty" in best_template:
        TEMPLATE_WEIGHTS[FLIRTY_KEY] = baseline + 0.5
    elif "coffee" in best_template:
        TEMPLATE_WEIGHTS[STRAIGHTFORWARD_KEY] = baseline + 0.5


def build_structured_profile_prompt() -> str:
    """
    Build the structured extraction prompt for the new main LLM call.
    The JSON format and allowed values are explicitly defined for clarity.
    """
    return (
        "You are analyzing screenshots of a dating profile. Each image may contain text, icons, or structured fields. "
        "Your task is to extract only the information that is explicitly visible on-screen. "
        "If a field is not directly stated, leave it empty (do not infer or guess).\n\n"
        "Return exactly one valid JSON object matching this structure, with the same field names, order, and formatting:\n\n"
        "{\n"
        '  "Name": "",\n'
        '  "Gender": "",\n'
        '  "Sexuality": "",\n'
        '  "Age": 0,  // Must be an integer\n'
        '  "Height": 0,  // Must be an integer\n'
        '  "Location": "",  // Pin icon\n'
        '  "Ethnicity": "",\n'
        '  "Children": "",  // Must be one of: "Don\'t have children", "Have children".\n'
        '  "Family plans": "",  // Must be one of: "Don\'t want children", "Want children", "Open to children", "Not sure yet".\n'
        '  "Covid Vaccine": "",  // Must be one of: "Vaccinated", "Partially vaccinated", "Not yet vaccinated".\n'
        '  "Pets": "",\n'
        '  "Zodiac Sign": "",\n'
        '  "Job title": "",\n'
        '  "University": "",\n'
        '  "Religious Beliefs": "",\n'
        '  "Home town": "",  // House icon\n'
        '  "Politics": "",\n'
        '  "Languages spoken": "",\n'
        '  "Dating Intentions": "",  // Must be one of: "Life partner", "Long-term relationship", "Long-term relationship, open to short", "Short-term relationship, open to long", "Short term relationship", "Figuring out my dating goals".\n'
        '  "Relationship type": "",  // Must be one of: "Monogamy", "Non-Monogamy", "Figuring out my relationship type".\n'
        '  "Drinking": "",  // Must be one of: "Yes", "Sometimes", "No".\n'
        '  "Smoking": "",  // Must be one of: "Yes", "Sometimes", "No".\n'
        '  "Marijuana": "",  // Must be one of: "Yes", "Sometimes", "No".\n'
        '  "Drugs": "",  // Must be one of: "Yes", "Sometimes", "No".\n'
        '  "Profile Prompts and Answers": [\n'
        '    {"prompt": "", "answer": ""},\n'
        '    {"prompt": "", "answer": ""},\n'
        '    {"prompt": "", "answer": ""}\n'
        '  ],\n'
        '  "Other text on profile not covered by above": "",\n'
        '  "Description of any non-photo media (For example poll, voice note)": "",\n'
        '  "Extensive Description of Photo 1": "",\n'
        '  "Extensive Description of Photo 2": "",\n'
        '  "Extensive Description of Photo 3": "",\n'
        '  "Extensive Description of Photo 4": "",\n'
        '  "Extensive Description of Photo 5": "",\n'
        '  "Extensive Description of Photo 6": "",\n'
        "}\n\n"
        "Rules:\n"
        "- Return only the JSON object, with no commentary, markdown, or code fences.\n"
        "- Do not include any text before or after the JSON.\n"
        "- Do not use synonyms or variations for categorical values.\n"
        "- If a field is not visible, leave it empty or null.\n"
        "- For 'Profile Prompts and Answers', extract three visible prompt/answer pairs.\n"
        "- For 'Other text', include any visible text that doesn’t fit the above categories (e.g., subtext, taglines, or captions).\n"
        "- One of the images will be a stitched image containing biometrics. This is part of the profile and should be included in the results.\n"
        "- For 'Extensive Description of Photo 1-6', you must provide a richly detailed description of the visible photos."
        "- For 'Description of any non-photo media (For example poll, voice note)', describe any visible non-photo media elements if present."
    )


def build_profile_eval_prompt(home_town: str, job_title: str, university: str) -> str:
    """
    Build the enrichment prompt for evaluating Home town, Job title, University.
    Returns a single prompt string instructing the model to output EXACTLY one JSON object.
    """
    parts = [
        "You are enriching structured dating profile fields for a scoring system. Use ONLY the provided text. Do not browse. Be conservative when uncertain.\n\n",
        "INPUT FIELDS (from the extracted JSON):\n",
        '- "Home town" (string; may be city/region/country or empty)\n',
        '- "Job title" (string; may be empty)\n',
        '- "University" (string; may be empty)\n\n',
        "VALUES:\n",
        f'Home town: "{home_town or ""}"\n',
        f'Job title: "{job_title or ""}"\n',
        f'University: "{university or ""}"\n\n',
        "YOUR TASKS (3):\n",
        '1) Resolve "Home town" to an ISO 3166-1 alpha-2 country code (uppercase). If it is a UK city/area (e.g., “Wembley”, “Harrow”, “Manchester”), return "GB".\n',
        '2) Map "Job title" to a London BASE salary band and typical salary estimate (exclude bonus/equity). If uncertain between two bands, choose the LOWER band and lower confidence.\n',
        '3) Check if "University" matches an elite list (case-insensitive), and return a 1/0 flag.\n\n',
        "IMPORTANT: Also compute and return the FINAL NUMERIC modifiers for each of these three dimensions, according to the exact rules below, so no additional processing is required.\n\n",
        "--------------------------------------------------------------------------------\n",
        "MODIFIER RULES (return these as numbers in the output):\n\n",
        "A) home_country_modifier\n",
        "- Use ISO country code from (1). Apply exactly ONE of the following:\n",
        "  +2 if ISO ∈ {NO, SE, DK, FI, IS, EE, LV, LT, UA, RU, BY}\n",
        "  +1 if ISO ∈ {IE, DE, FR, NL, BE, LU, CH, AT, IT, ES, PT, PL, CZ, CA, US, AU, NZ, JP, KR, SG, IL, AE}\n",
        "  -1 if ISO is in South America or Southeast Asia (lists below), EXCEPT SG is never penalized\n",
        "   South America ISO list: {AR, BR, CL, CO, PE, EC, UY, PY, BO, VE, GY, SR}\n",
        "   Southeast Asia ISO list: {ID, MY, TH, VN, PH, KH, LA, MM, BN}\n",
        'Else 0 if ISO is "GB" or unresolved/empty.\n',
        'If unresolved: home_country_iso = "" and home_country_modifier = 0.\n\n',
        "B) job_modifier (derived from job band with overrides)\n",
        "- First assign a band using London BASE salary only (no bonus/equity):\n",
        "  B1: < 30000\n",
        "  B2: 30000–60000\n",
        "  B3: 60000–100000\n",
        "  B4: 100000–180000\n",
        "  B5: > 180000\n",
        "- Then set job_modifier from band:\n",
        "  B1 → -1, B2 → 0, B3 → +1, B4 → +2, B5 → +3\n",
        "- OVERRIDES:\n",
        '  - If title contains "Student", "Intern", or "Graduate" (case-insensitive): job_modifier = 0 (even if band would be higher).\n',
        '  - If title contains "PhD" or "Doctoral": the band is B3 for scoring purposes (job_modifier = +1).\n',
        "  - If you are uncertain (confidence < 0.4), reduce job_modifier by 1 step (minimum -1).\n",
        '  - If title is empty or cannot be resolved: normalized_title = "Unknown", est_salary_gbp = 0, band = "B2", confidence = 0.2, job_modifier = 0.\n\n',
        "C) university_modifier\n",
        "- Elite universities list (case-insensitive exact name match after trimming):\n",
        '  ["University of Oxford","University of Cambridge","Imperial College London","London School of Economics","Harvard University","Yale University","Princeton University","Stanford University","MIT","Columbia University","ETH Zürich","EPFL","University of Copenhagen","Sorbonne University","University of Tokyo","National University of Singapore","Tsinghua University","Peking University","University of Toronto","Australian National University","University of Melbourne","University of Hong Kong"]\n',
        "- university_elite = 1 if matched, else 0\n",
        "- university_modifier = +1 if university_elite = 1, else 0\n",
        '- matched_university_name = the canonical elite name matched, else "".\n\n',
        "--------------------------------------------------------------------------------\n",
        "OUTPUT EXACTLY ONE JSON OBJECT (no commentary, no code fences):\n\n",
        "{\n",
        '  "home_country_iso": "",           // ISO alpha-2 or ""\n',
        '  "home_country_confidence": 0.0,   // 0.0-1.0\n',
        '  "home_country_modifier": 0,       // final numeric modifier per rules above\n\n',
        '  "job": {\n',
        '    "normalized_title": "",         // concise title or "Unknown"\n',
        '    "est_salary_gbp": 0,            // integer, base salary typical for LONDON\n',
        '    "band": "",                     // "B1" | "B2" | "B3" | "B4" | "B5"\n',
        '    "confidence": 0.0,              // 0.0-1.0\n',
        '    "band_reason": ""               // one short sentence justifying the band\n',
        "  },\n",
        '  "job_modifier": 0,                // final numeric modifier after overrides/uncertainty\n\n',
        '  "university_elite": 0,            // 1 or 0\n',
        '  "matched_university_name": "",\n',
        '  "university_modifier": 0          // +1 if elite else 0\n',
        "}\n",
    ]
    return "".join(parts)


if __name__ == "__main__":
    # For quick verification
    print(build_structured_profile_prompt())
