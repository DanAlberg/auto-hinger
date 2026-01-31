from typing import Any, Dict, List

from profile_utils import _get_core, _get_visual, _norm_value, _split_csv

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

    chest = visual_val("Apparent Chest Proportions")
    chest_norm = _norm_value(chest)
    if chest_norm == _norm_value("Petite/small/narrow"):
        record("Visual Analysis", "Apparent Chest Proportions", chest, -5)
    elif chest_norm and chest_norm != _norm_value("Average/balanced/proportional"):
        record("Visual Analysis", "Apparent Chest Proportions", chest, +5)

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

    chest = visual_val("Apparent Chest Proportions")
    chest_norm = _norm_value(chest)
    if chest_norm == _norm_value("Petite/small/narrow"):
        record("Visual Analysis", "Apparent Chest Proportions", chest, -5)
    elif chest_norm and chest_norm != _norm_value("Average/balanced/proportional"):
        record("Visual Analysis", "Apparent Chest Proportions", chest, +5)

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
