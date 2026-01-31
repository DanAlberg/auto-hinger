import json
from typing import Any, Dict

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
        '    "Apparent Chest Proportions": "",\n'
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
        "- Provide a detailed, visual summary of the main subject: clothing, pose, activity, background, "
        "facial features when visible, skin tone, build, grooming, accessories, and overall presentation. "
        "Unbiased, accurate responses are required. Conventionally unattractive people should be labelled as so, false positivity should be avoided.\n\n"
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
        '"Apparent Chest Proportions": "Petite/small/narrow", "Average/balanced/proportional", "Defined/toned", "Full/curvy", "Prominent/voluptuous", "Broad/strong"\n'
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
        "These should be warm, natural, and genuinely curious questions — the kind of thing a normal, confident guy would send.\n"
        "The goal is to start an easy conversation.\n\n"
        "Profile details:\n"
        f"{extracted_json}\n\n"
        "Task: Generate exactly 10 DISTINCT opening messages as JSON.\n\n"
        "Rules:\n"
        "- Output JSON only, no extra text.\n"
        "- Each opener must anchor to EXACTLY ONE profile element ID: prompt_1..prompt_3, photo_1..photo_6, poll_1_a|poll_1_b|poll_1_c.\n"
        "- Every opener must be a simple question (or light A/B choice) that is easy to reply to.\n"
        "- Keep it sweet, and either curious or flirty\n"
        "- Do NOT narrate the photo or prompt like a caption. Reference it naturally (\"where was this?\" not \"in photo 3\" or \"that rooftop looks...\"). The text will be linked to the photo by the user, using the index supplied, and the recipient will know which photo you were referring to. Act as if in natural conversation\n"
        "- Messages should feel human, relaxed, and effortless.\n\n"
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
        "- Do NOT narrate the photo or prompt like a caption. Reference it naturally (\"where was this?\" not \"in photo 3\" or \"that rooftop looks...\"). The text will be linked to the photo by the user, using the index supplied, and the recipient will know which photo you were referring to. Act as if in natural conversation\n"
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
        "Pick the one that is the best, least cringy, and most importantly most likely to get a reply. Be decisive.\n\n"
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



