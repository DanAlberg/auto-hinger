import os
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any, Tuple


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def get_db_path() -> str:
    return os.path.join(_repo_root(), "profiles.db")


def init_db(db_path: Optional[str] = None) -> None:
    """
    Initialize the SQLite database with WAL mode and the flattened profiles table.
    Schema mirrors build_structured_profile_prompt (extracted) + build_profile_eval_prompt (enrichment) + score.
    Dedup is enforced via UNIQUE(Name COLLATE NOCASE, Age, Height_cm).
    """
    db_path = db_path or get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        # WAL for read-while-write; NORMAL synchronous for perf
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")

        # Main flattened table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY,
                -- Core extracted fields
                Name TEXT NOT NULL,
                Gender TEXT,
                Sexuality TEXT,
                Age INTEGER NOT NULL,
                Height_cm INTEGER NOT NULL,
                Location TEXT,
                Ethnicity TEXT,
                Children TEXT,
                Family_plans TEXT,
                Covid_vaccine TEXT,
                Pets TEXT,
                Zodiac_Sign TEXT,
                Job_title TEXT,
                University TEXT,
                Religious_Beliefs TEXT,
                Home_town TEXT,
                Politics TEXT,
                Languages_spoken TEXT,
                Dating_Intentions TEXT,
                Relationship_type TEXT,
                Drinking TEXT,
                Smoking TEXT,
                Marijuana TEXT,
                Drugs TEXT,
                prompt_1 TEXT,
                answer_1 TEXT,
                prompt_2 TEXT,
                answer_2 TEXT,
                prompt_3 TEXT,
                answer_3 TEXT,
                Other_text TEXT,
                Media_description TEXT,
                Photo1_desc TEXT,
                Photo2_desc TEXT,
                Photo3_desc TEXT,
                Photo4_desc TEXT,
                Photo5_desc TEXT,
                Photo6_desc TEXT,
                -- Enrichment (profile_eval)
                home_country_iso TEXT,
                home_country_confidence REAL,
                home_country_modifier INTEGER,
                job_normalized_title TEXT,
                job_est_salary_gbp INTEGER,
                job_band TEXT,
                job_confidence REAL,
                job_band_reason TEXT,
                job_modifier INTEGER,
                university_elite INTEGER,
                matched_university_name TEXT,
                university_modifier INTEGER,
                -- Derived
                score INTEGER,
                score_breakdown TEXT,
                timestamp TEXT
            );
            """
        )

        # Dedup: case-insensitive name + age + height_cm
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_profiles_unique
            ON profiles(Name COLLATE NOCASE, Age, Height_cm);
            """
        )
        # Ensure score_breakdown column exists on pre-existing DBs
        try:
            cur.execute("ALTER TABLE profiles ADD COLUMN score_breakdown TEXT;")
        except Exception:
            pass

        con.commit()
    finally:
        con.close()


# ---------------------- Flatten helpers ----------------------

def _coerce_int(val: Any, field: str) -> int:
    try:
        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, str):
            s = val.strip()
            if s == "":
                raise ValueError
            return int(float(s))
    except Exception:
        raise ValueError(f"{field} must be an integer; got {val!r}")
    raise ValueError(f"{field} must be an integer; got {val!r}")


def _extract_prompts(extracted: Dict[str, Any]) -> Tuple[str, str, str, str, str, str]:
    """
    Map 'Profile Prompts and Answers' list to prompt_1/answer_1 .. prompt_3/answer_3
    """
    p1 = a1 = p2 = a2 = p3 = a3 = ""
    arr = extracted.get("Profile Prompts and Answers") or extracted.get("profile_prompts_and_answers") or []
    if isinstance(arr, list):
        if len(arr) > 0 and isinstance(arr[0], dict):
            p1 = (arr[0].get("prompt") or "").strip()
            a1 = (arr[0].get("answer") or "").strip()
        if len(arr) > 1 and isinstance(arr[1], dict):
            p2 = (arr[1].get("prompt") or "").strip()
            a2 = (arr[1].get("answer") or "").strip()
        if len(arr) > 2 and isinstance(arr[2], dict):
            p3 = (arr[2].get("prompt") or "").strip()
            a3 = (arr[2].get("answer") or "").strip()
    return p1, a1, p2, a2, p3, a3


def _val(d: Dict[str, Any], key1: str, key2: Optional[str] = None) -> Any:
    if key1 in d:
        return d.get(key1)
    if key2 and key2 in d:
        return d.get(key2)
    # case-insensitive fallback
    lower_map = {k.lower(): k for k in d.keys()}
    if key1.lower() in lower_map:
        return d.get(lower_map[key1.lower()])
    if key2 and key2.lower() in lower_map:
        return d.get(lower_map[key2.lower()])
    return None


def _flatten_extracted(extracted: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten the structured profile JSON into flat columns.
    Height is required and becomes Height_cm (assumed already centimeters).
    Age is required integer.
    """
    if not isinstance(extracted, dict):
        extracted = {}

    name = _val(extracted, "Name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Name is required")

    age_raw = _val(extracted, "Age")
    age = _coerce_int(age_raw, "Age")
    height_raw = _val(extracted, "Height")
    height_cm = _coerce_int(height_raw, "Height")

    (prompt_1, answer_1, prompt_2, answer_2, prompt_3, answer_3) = _extract_prompts(extracted)

    row = {
        "Name": name.strip(),
        "Gender": _val(extracted, "Gender") or "",
        "Sexuality": _val(extracted, "Sexuality") or "",
        "Age": age,
        "Height_cm": height_cm,
        "Location": _val(extracted, "Location") or "",
        "Ethnicity": _val(extracted, "Ethnicity") or "",
        "Children": _val(extracted, "Children") or "",
        "Family_plans": _val(extracted, "Family plans") or _val(extracted, "Family_plans") or "",
        "Covid_vaccine": _val(extracted, "Covid Vaccine") or _val(extracted, "Covid_vaccine") or "",
        "Pets": _val(extracted, "Pets") or "",
        "Zodiac_Sign": _val(extracted, "Zodiac Sign") or _val(extracted, "Zodiac_Sign") or "",
        "Job_title": _val(extracted, "Job title") or _val(extracted, "Job_title") or "",
        "University": _val(extracted, "University") or "",
        "Religious_Beliefs": _val(extracted, "Religious Beliefs") or _val(extracted, "Religious_Beliefs") or "",
        "Home_town": _val(extracted, "Home town") or _val(extracted, "Home_town") or "",
        "Politics": _val(extracted, "Politics") or "",
        "Languages_spoken": _val(extracted, "Languages spoken") or _val(extracted, "Languages_spoken") or "",
        "Dating_Intentions": _val(extracted, "Dating Intentions") or _val(extracted, "Dating_Intentions") or "",
        "Relationship_type": _val(extracted, "Relationship type") or _val(extracted, "Relationship_type") or "",
        "Drinking": _val(extracted, "Drinking") or "",
        "Smoking": _val(extracted, "Smoking") or "",
        "Marijuana": _val(extracted, "Marijuana") or "",
        "Drugs": _val(extracted, "Drugs") or "",
        "prompt_1": prompt_1,
        "answer_1": answer_1,
        "prompt_2": prompt_2,
        "answer_2": answer_2,
        "prompt_3": prompt_3,
        "answer_3": answer_3,
        "Other_text": _val(extracted, "Other text on profile not covered by above") or "",
        "Media_description": _val(extracted, "Description of any non-photo media (For example poll, voice note)") or "",
        "Photo1_desc": _val(extracted, "Extensive Description of Photo 1") or "",
        "Photo2_desc": _val(extracted, "Extensive Description of Photo 2") or "",
        "Photo3_desc": _val(extracted, "Extensive Description of Photo 3") or "",
        "Photo4_desc": _val(extracted, "Extensive Description of Photo 4") or "",
        "Photo5_desc": _val(extracted, "Extensive Description of Photo 5") or "",
        "Photo6_desc": _val(extracted, "Extensive Description of Photo 6") or "",
    }
    return row


def _flatten_enrichment(enrichment: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(enrichment, dict):
        enrichment = {}
    job = enrichment.get("job") or {}
    return {
        "home_country_iso": enrichment.get("home_country_iso") or "",
        "home_country_confidence": float(enrichment.get("home_country_confidence") or 0.0),
        "home_country_modifier": int(enrichment.get("home_country_modifier") or 0),
        "job_normalized_title": job.get("normalized_title") or "",
        "job_est_salary_gbp": int(job.get("est_salary_gbp") or 0),
        "job_band": job.get("band") or "",
        "job_confidence": float(job.get("confidence") or 0.0),
        "job_band_reason": job.get("band_reason") or "",
        "job_modifier": int(enrichment.get("job_modifier") or 0),
        "university_elite": int(enrichment.get("university_elite") or 0),
        "matched_university_name": enrichment.get("matched_university_name") or "",
        "university_modifier": int(enrichment.get("university_modifier") or 0),
    }


# ---------------------- Upsert API ----------------------

def upsert_profile_flat(
    extracted_profile: Dict[str, Any],
    enrichment: Dict[str, Any],
    score: int,
    score_breakdown: Optional[str] = None,
    timestamp: Optional[str] = None,
    db_path: Optional[str] = None
) -> Optional[int]:
    """
    Flatten and UPSERT a profile row.
    - Enforces UNIQUE(Name NOCASE, Age, Height_cm) with ON CONFLICT DO NOTHING.
    - Returns rowid on insert; None if conflict (duplicate).
    - Height and Age required; ValueError raised otherwise.
    """
    db_path = db_path or get_db_path()
    init_db(db_path)  # safe to call repeatedly

    core = _flatten_extracted(extracted_profile or {})
    enrich = _flatten_enrichment(enrichment or {})
    ts = timestamp or datetime.now().isoformat(timespec="seconds")

    row = {
        **core,
        **enrich,
        "score": int(score),
        "score_breakdown": (score_breakdown or ""),
        "timestamp": ts,
    }

    # Build insert with named parameters
    cols = [
        "Name","Gender","Sexuality","Age","Height_cm","Location","Ethnicity","Children","Family_plans",
        "Covid_vaccine","Pets","Zodiac_Sign","Job_title","University","Religious_Beliefs","Home_town",
        "Politics","Languages_spoken","Dating_Intentions","Relationship_type","Drinking","Smoking",
        "Marijuana","Drugs","prompt_1","answer_1","prompt_2","answer_2","prompt_3","answer_3",
        "Other_text","Media_description","Photo1_desc","Photo2_desc","Photo3_desc","Photo4_desc","Photo5_desc","Photo6_desc",
        "home_country_iso","home_country_confidence","home_country_modifier","job_normalized_title","job_est_salary_gbp",
        "job_band","job_confidence","job_band_reason","job_modifier","university_elite","matched_university_name","university_modifier",
        "score","score_breakdown","timestamp"
    ]
    placeholders = ",".join([f":{c}" for c in cols])
    col_list = ",".join(cols)

    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            f"""
            INSERT INTO profiles ({col_list})
            VALUES ({placeholders})
            ON CONFLICT(Name, Age, Height_cm) DO NOTHING;
            """,
            row
        )
        con.commit()
        if cur.rowcount == 0:
            return None
        return cur.lastrowid
    finally:
        con.close()
