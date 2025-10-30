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


# System prompt for choosing opener style; JSON-only response required.
OPENING_STYLE_SYSTEM_PROMPT = (
    "You are a dating-opener strategist. Given a structured profile, decide what kind of opener would work best "
    "on this person and explain why. Think like someone who actually uses dating apps and understands attraction, tone, and humour. "
    "Be direct, perceptive, and conversational rather than cautious or corporate.\n\n"
    "Photo descriptions may sound factual or neutral. Infer the social tone and vibe behind them — for example, if someone is "
    "making eye contact, laughing, posing confidently, or dressed for nightlife, that can imply flirty, playful, or bold energy. "
    "Use normal human language like confident, flirty, cheeky, relaxed, or reserved where appropriate.\n\n"
    "Task\n"
    "Pick the single primary_style from this set:\n"
    "[flirty, complimentary, playful_witty, observational, shared_interest, genuinely_warm, relationship_forward]\n\n"
    "Provide style_weights for all styles (0–1, sum to 1).\n"
    "Give an overall_confidence (0–1).\n"
    "Provide a short rationale (≤120 words) describing the personality and tone you infer from the profile and why that opener style fits best. "
    "Reference concrete cues from prompts, phrasing, or photos.\n\n"
    "Output (JSON only)\n"
    "{\n"
    '  "primary_style": "observational",\n'
    '  "style_weights": {\n'
    '    "flirty": 0.25,\n'
    '    "complimentary": 0.10,\n'
    '    "playful_witty": 0.20,\n'
    '    "observational": 0.25,\n'
    '    "shared_interest": 0.10,\n'
    '    "genuinely_warm": 0.05,\n'
    '    "relationship_forward": 0.05\n'
    "  },\n"
    '  "overall_confidence": 0.74,\n'
    '  "rationale": "She gives off confident, cheeky energy through her humour and nightlife photos, so playful_witty leads with some flirty overlap."\n'
    "}\n\n"
    "Scoring guidance (internal)\n"
    "• Increase flirty if photos or prompts show confidence, eye contact, nightlife, or teasing humour.\n"
    "• Increase complimentary if they seem warm, elegant, expressive, or clearly enjoy positive attention.\n"
    "• Increase playful_witty when there’s irony, self-aware humour, or offbeat storytelling.\n"
    "• Increase observational when visuals, props, or context (venues, outfits, scenes) reveal personality cues worth riffing on.\n"
    "• Increase shared_interest when hobbies, travel, or niche passions are central.\n"
    "• Increase genuinely_warm when tone feels kind, grounded, or low-key.\n"
    "• Increase relationship_forward when long-term intent or values are clearly stated.\n"
    "• If the profile’s humour or imagery feels bold, teasing, or sexually confident, bias upward for flirty or playful even if text is restrained.\n"
    "• Keep overall_confidence moderate when tone signals conflict or ambiguity."
)



def _line(label: str, value: str) -> str:
    v = (value or "").strip()
    return f"{label}: {v}" if v else ""


def render_opening_style_user_message(profile: dict) -> str:
    """
    Render the user message using ONLY separate lines (no combined fields).
    Use normal dashes for prompt/answer joins if needed; however, prompts and answers
    are rendered as separate lines to avoid combining.
    Only non‑empty values are included.
    """
    p = profile or {}

    # Height formatting (from Height_cm)
    height_cm = p.get("Height_cm")
    height_str = ""
    try:
        if height_cm is not None and str(height_cm).strip() != "":
            height_val = int(float(height_cm))
            if height_val > 0:
                height_str = f"{height_val} cm"
    except Exception:
        pass

    lines = []
    lines.append("Profile:")

    # Core identity
    for (label, key) in [
        ("Name", "Name"),
        ("Age", "Age"),
        ("Gender", "Gender"),
        ("Sexuality", "Sexuality"),
        ("Location", "Location"),
        ("Home town", "Home_town"),
        ("Ethnicity", "Ethnicity"),
        ("Job title", "Job_title"),
        ("University", "University"),
        ("Religious beliefs", "Religious_Beliefs"),
        ("Politics", "Politics"),
        ("Languages spoken", "Languages_spoken"),
        ("Dating intention", "Dating_Intentions"),
        ("Relationship type", "Relationship_type"),
        ("Drinking", "Drinking"),
        ("Smoking", "Smoking"),
        ("Marijuana", "Marijuana"),
        ("Drugs", "Drugs"),
        ("Children", "Children"),
        ("Family plans", "Family_plans"),
        ("Pets", "Pets"),
    ]:
        line = _line(label, str(p.get(key, "")).strip())
        if line:
            lines.append(line)

    # Height as its own line
    if height_str:
        lines.append(f"Height: {height_str}")

    # Prompts & answers: assume present; include light safeguard
    lines.append("Prompts & answers:")
    for i in (1, 2, 3):
        prompt_i = str(p.get(f"prompt_{i}", "") or "").strip()
        answer_i = str(p.get(f"answer_{i}", "") or "").strip()
        if prompt_i:
            lines.append(f"prompt_{i}: {prompt_i}")
        if answer_i:
            lines.append(f"answer_{i}: {answer_i}")

    # Other text
    other_text = (p.get("Other_text") or "").strip()
    if other_text:
        lines.append(f"Other text: {other_text}")

    # Photo/context bullets
    photo_keys = [
        ("Photo1", "Photo1_desc"),
        ("Photo2", "Photo2_desc"),
        ("Photo3", "Photo3_desc"),
        ("Photo4", "Photo4_desc"),
        ("Photo5", "Photo5_desc"),
        ("Photo6", "Photo6_desc"),
    ]
    any_photos = any((p.get(k) or "").strip() for _, k in photo_keys)
    if any_photos:
        lines.append("Photo/context (short bullet summaries):")
        for label, key in photo_keys:
            desc = (p.get(key) or "").strip()
            if desc:
                lines.append(f"{label}: {desc}")

    # Media summary
    media = (p.get("Media_description") or "").strip()
    if media:
        lines.append(f"Media summary: {media}")

    return "\n".join(lines)


def build_opening_style_prompts(profile: dict) -> tuple[str, str]:
    """
    Returns (system, user) prompts for the opening‑style request.
    """
    return OPENING_STYLE_SYSTEM_PROMPT, render_opening_style_user_message(profile)


# ---------------- Opening messages (British English, JSON-only) ----------------
OPENING_MESSAGES_SYSTEM_PROMPT = (
    "You are a dating-app message generator. Write like a confident, witty man in his mid-20s who knows how to flirt "
    "and hold a woman’s attention. Your goal is to create short, engaging opening messages tailored to the other person’s profile "
    "and a style analysis. Be charming, observant, and naturally flirty when the profile allows. "
    "Use British English spelling and punctuation. Avoid anything that sounds formal, corporate, or over-safe."
    "Encoding: ASCII punctuation only. Do not output Unicode dashes (U+2013 or U+2014). Use a normal hyphen (-), full stop, or comma instead."

)


def build_opening_messages_prompts(profile_json: dict, opening_style_json: dict) -> tuple[str, str]:
    """
    Build (system, user) prompts for generating 10 opening messages.
    Automatically includes Daniel's persona for tone grounding and overlap awareness.
    """
    import json

    sender_persona = {
        "name": "Daniel",
        "vibe": "confident, witty, flirtatious, well-educated",
        "university": "UCL",
        "occupation": "Head of AI Security (AI and Cybersecurity)"
    }

    prof_str = json.dumps(profile_json or {}, ensure_ascii=False, indent=2)
    style_str = json.dumps(opening_style_json or {}, ensure_ascii=False, indent=2)
    persona_str = json.dumps(sender_persona, ensure_ascii=False, indent=2)

    user = (
        f"Sender persona:\n{persona_str}\n\n"
        f"Profile (scraped):\n{prof_str}\n\n"
        f"Style analysis:\n{style_str}\n\n"
        "Task\n\n"
        "Generate 10 unique opening messages that maximise reply likelihood and fit the personality and tone of this profile. "
        "Each message must be inspired by one specific photo or prompt, which you must identify for targeting.\n\n"
        "Tone and intent\n"
        "• Write like a socially fluent man who’s attractive, confident, and intelligent.\n"
        "• Be flirty, teasing, or curious when it feels natural; don’t play it safe.\n"
        "• Use humour and light provocation over politeness.\n"
        "• Never over-explain what you’re referencing — sound like someone chatting, not demonstrating understanding.\n"
        "• You may reference overlap between Daniel and the recipient if it feels natural and adds intrigue "
        "When referring to photos, use natural phrasing like “that rooftop” or “that outfit”, never “in the red-lit photo” or “in photo 2”."
        "Avoid cliche phrases or facts. Be original and obscure."
        "(for example, same university or similar field). Never list credentials or facts — it should sound like spontaneous chemistry, not a CV.\n\n"
        "Structure\n"
        "• Exactly 10 messages:\n"
        "  – 3 written in the primary style.\n"
        "  – 2 written in the second-highest weighted style.\n"
        "  – 5 synthesised, blending tone and intent across all style weights.\n\n"
        "• Length: 1–2 sentences per message. No em dashes.\n"
        "• Keep language modern and natural; mid-20s tone, not overly slangy or American.\n"
        "• A/B or short question formats are fine if they flow naturally.\n"
        "• You can use well-known London area names if they make sense for a date or activity suggestion, "
        "but never mention where she lives, studies, or works.\n"
        "• Grounding should be implicit — the line can be inspired by a photo or prompt but doesn’t need to describe it.\n"
        "• No mechanical prompt+photo mashups.\n"
        "• Every opener must feel distinct in tone and intent.\n\n"
        "Output (JSON only)\n"
        "{\n"
        '  "openers": [\n'
        "    {\n"
        '      "text": "Opening line here.",\n'
        '      "intended_style": "playful_witty",\n'
        '      "target_type": "photo",\n'
        '      "target_index": 2,\n'
        '      "target_summary": "rooftop skyline at night",\n'
        '      "hook_basis": "nightlife vibe and smile",\n'
        '      "reply_affordance": 0.86\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Field notes\n"
        "text – the message to send.\n"
        "intended_style – one of [flirty, complimentary, playful_witty, observational, shared_interest, genuinely_warm, relationship_forward].\n"
        "target_* – identifies the element it’s responding to (for programmatic liking).\n"
        "hook_basis – short note of what inspired it (not shown to user).\n"
        "reply_affordance – 0–1 model estimate of how easy it is to reply.\n"
    )
    return OPENING_MESSAGES_SYSTEM_PROMPT, user



# ---------------- Opening pick (choose best opener; British English, JSON-only) ----------------
OPENING_PICK_SYSTEM_PROMPT = (
    "You are a dating-app strategist. Choose the single best opening line from a generated list. "
    "Judge like someone who understands attraction, wit, and conversational chemistry — not a marketer. "
    "Favour confidence, intrigue, and personality over politeness or caution. "
    "Use British English spelling and punctuation."
)


def build_opening_pick_prompts(profile_json: dict, opening_style_json: dict, generated_openers_json: dict) -> tuple[str, str]:
    """
    Build (system, user) prompts for selecting the single best opening line.
    """
    import json
    prof_str = json.dumps(profile_json or {}, ensure_ascii=False, indent=2)
    style_str = json.dumps(opening_style_json or {}, ensure_ascii=False, indent=2)
    gens_str = json.dumps(generated_openers_json or {}, ensure_ascii=False, indent=2)

    user = (
        "Profile (scraped):\n\n"
        f"{prof_str}\n\n\n"
        "Style analysis:\n\n"
        f"{style_str}\n\n\n"
        "Candidate openers (from previous step):\n\n"
        f"{gens_str}\n\n"
        "Task\n\n"
        "Select one opening line that is most likely to:\n"
        "• Catch her attention immediately among many messages.\n"
        "• Create intrigue or attraction — it should feel confident and engaging.\n"
        "• Be effortless to reply to (simple, fun, or flirty).\n"
        "• Match her tone and personality based on the style analysis.\n"
        "• Sound like it came from a socially fluent, intelligent man — not a bot, not a try-hard.\n\n"
        "Constraints\n\n"
        "• You must choose exactly one of the ten.\n"
        "• Keep justification short (1–2 sentences) and concrete.\n"
        "• Weigh style alignment, intrigue, and ease of reply equally — don’t just pick the safest one.\n"
        "• Ground reasoning in actual wording, tone, and her profile cues.\n"
        "• Use British English.\n\n"
        "Output JSON only.\n\n"
        "Output format\n"
        "{\n"
        '  "chosen_text": "Final selected opening line here.",\n'
        '  "chosen_index": 7,\n'
        '  "intended_style": "playful_witty",\n'
        '  "target_type": "photo",\n'
        '  "target_index": 3,\n'
        '  "target_summary": "Photo laughing in a red floral dress at a wine bar",\n'
        '  "justification": "It’s confident, playful, and easy to answer — it fits her humour and feels natural, not rehearsed.",\n'
        '  "confidence": 0.84\n'
        "}\n"
    )
    return OPENING_PICK_SYSTEM_PROMPT, user



if __name__ == "__main__":
    # For quick verification
    print(build_structured_profile_prompt())
