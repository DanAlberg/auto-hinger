import time
from datetime import datetime
from profile_export import ProfileExporter, SCHEMA_V2
import sqlite3

def build_sample_row():
    # Provide reasonable demo values for scan-derived fields; leave others blank
    defaults = {
        "name": "Test Person",
        "age": 28,
        "height": "5'7\"",
        "location": "London",
        "sexuality": "",
        "ethnicity": "",
        "current_children": "",
        "family_plans": "Open to children",
        "covid_vaccine": "",
        "zodiac_sign": "",
        "hometown": "",
        "university": "UCL",
        "job_title": "Engineer",
        "work": "Acme Corp",
        "religious_beliefs": "",
        "politics": "Moderate",
        "languages_spoken": "English, Spanish",
        "dating_intentions": "",
        "relationship_type": "",
        "drinking": "Sometimes",
        "smoking": "No",
        "marijuana": "No",
        "drugs": "No",
        "pets_dog": 1,
        "pets_cat": 0,
        "pets_bird": "",
        "pets_fish": "",
        "pets_reptile": "",
        "bio": "Sample bio derived from screenshots.",
        "prompts_and_answers": "My simple pleasures: Coffee; Best travel story: Got lost in Kyoto",
        "interests": "Climbing, Photography",
        "summary": "Friendly, active, loves coffee.",
    }
    # Ensure row contains all schema keys in order; empty string for any missing
    return {key: defaults.get(key, "") for key in SCHEMA_V2}

def main():
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    exporter = ProfileExporter(export_dir=".", session_id=session_id)
    row = build_sample_row()
    exporter.append_row(row)
    # Attempt duplicate insert to verify de-duplication by Name+Age+Height
    exporter.append_row(row)
    paths = exporter.get_paths()
    db_path = paths.get("db")
    print("Wrote row to DB:", db_path)
    # Verify only 1 row recorded for this session_id
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM profiles WHERE session_id = ?", (session_id,))
        count = cur.fetchone()[0]
        print("Rows for this session_id:", count)
        if count != 1:
            print("⚠️ Dedup check unexpected count:", count)
        else:
            print("✅ Dedup check passed (1 row for duplicate inserts)")
    finally:
        try:
            con.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
