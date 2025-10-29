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
        '1) Resolve "Home town" to an ISO 3166-1 alpha-2 country code (uppercase). If it is a UK city/area (e.g., “Wembley”, “Harrow”, “Manchester”), return "GB".\n',
        '2) Estimate FUTURE EARNING POTENTIAL (TIER) from the vague job/study field AND the university context. Titles are often minimal (e.g., “Tech”, “Finance”, “Product”, “Student”, “PhD”). Use the tier table in section B and return the corresponding numeric modifier as job_modifier. When uncertain between two adjacent tiers, be slightly optimistic and choose the higher tier by at most one step.\n',
        '3) Check if "University" matches an elite list (case-insensitive), and return a 1/0 flag.\n\n',
        "IMPORTANT: Also compute and return the FINAL NUMERIC modifiers for these dimensions so no additional processing is required.\n\n",
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
        "B) FUTURE EARNING POTENTIAL (tiers T0–T4) → job_modifier\n",
        "- Goal: infer likely earning trajectory within ~10 years using BOTH job/study field and university context (if visible). Classify into one of these tiers and set job.band = \"T0\"–\"T4\" accordingly:\n",
        "  T0 (modifier -2): Low/no trajectory. Clear low-mobility sectors with low ceiling and no elite cues: retail, hospitality, customer service, basic admin, charity/NGO support, nanny/TA, generic creative with no domain anchor.\n",
        "  T1 (modifier 0): Stable but capped. Teacher, nurse, social worker, marketing/HR/recruitment/ops/comms, public-sector researcher, therapist/psychology, non-STEM PhD, generic “research”.\n",
        "  T2 (modifier +2): Mid/high potential. Engineer, analyst, product manager, consultant, doctor, solicitor, finance, data, law, scientist, sales, generic “tech/software/PM”, STEM PhD, or STUDENT with elite STEM context.\n",
        "  T3 (modifier +4): High trajectory. Investment/banking, management consulting, quant, strategy, PE/VC, corporate law (Magic Circle), AI/data scientist, Big-Tech-calibre product/engineering, “Head/Lead/Director” (early leadership cues).\n",
        "  T4 (modifier +6): Exceptional (rare). Partner/Principal/Director (large firm), VP, funded founder with staff, senior specialist physicians, staff/principal engineer. Require strong textual cues.\n",
        "- Beneficial-doubt rule for missing or humorous titles: If the job field is empty or clearly humorous (e.g., “Glorified babysitter”), assign **T1 by default**, and upgrade to **T2** if elite-STEM education or strong sector hints justify it.\n",
        "- University influence: If elite uni AND STEM/quant field hints, allow T2–T3 even for “Student/PhD”. If elite uni but non-STEM, at most T1–T2 unless sector hints justify higher.\n",
        "- Vague sector keyword mapping (examples, not exhaustive):\n",
        "   “Tech/Software/Engineer/Data/PM/AI” → T2; consider T3 with elite context.\n",
        "   “Finance/Banking/Investment/Analyst” → T2; consider T3 with elite context.\n",
        "   “Consulting/Strategy” → T2; consider T3 with elite context.\n",
        "   “Law/Solicitor/Legal” → T2; consider T3 with Magic Circle/elite context.\n",
        "   “Marketing/HR/Recruitment/Ops/Comms/Education/Therapy/Research” → T1 by default; upgrade to T2 only with strong signals.\n",
        "- Confidence: return confidence 0.0–1.0 for the chosen tier. Do NOT downscale the tier due to low confidence; the optimism rule already limits to a one-step upgrade.\n",
        "- Output **job_modifier** directly from the chosen tier using the mapping above.\n\n",
        "C) university_modifier\n",
        "- Elite universities list (case-insensitive exact name match after trimming):\n",
        '  ["University of Oxford","University of Cambridge","Imperial College London", "UCL", "London School of Economics","Harvard University","Yale University","Princeton University","Stanford University","MIT","Columbia University","ETH Zürich","EPFL","University of Copenhagen","Sorbonne University","University of Tokyo","National University of Singapore","Tsinghua University","Peking University","University of Toronto","Australian National University","University of Melbourne","University of Hong Kong"]\n',
        "- Matching rule: If the University field contains multiple names or partial mentions (e.g., \"Oxford, PhD @ UCL\"), treat it as elite if ANY part contains an elite name (case-insensitive). Set matched_university_name to the canonical elite name.\n",
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
        '    "normalized_title": "",         // concise title or \"Unknown\"\n',
        '    "est_salary_gbp": 0,            // keep 0 if not inferred; not used for scoring in this scheme\n',
        '    "band": "",                     // USE \"T0\"|\"T1\"|\"T2\"|\"T3\"|\"T4\" to denote future earning tier\n',
        '    "confidence": 0.0,              // 0.0-1.0 confidence for the chosen tier\n',
        '    "band_reason": ""               // one short sentence justifying the tier choice\n',
        "  },\n",
        '  "job_modifier": 0,                // final numeric modifier from the tier mapping\n\n',
        '  "university_elite": 0,            // 1 or 0\n',
        '  "matched_university_name": "",\n',
        '  "university_modifier": 0          // +1 if elite else 0\n',
        "}\n",
    ]
    return "".join(parts)



if __name__ == "__main__":
    # For quick verification
    print(build_structured_profile_prompt())
