import time
from datetime import datetime
from profile_export import ProfileExporter, SCHEMA_V2

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
    exporter = ProfileExporter(export_dir=".", session_id=session_id, export_xlsx=True)
    row = build_sample_row()
    exporter.append_row(row)
    paths = exporter.get_paths()
    print("Wrote row to:", paths.get("xlsx"))

if __name__ == "__main__":
    # Allow time to open the workbook in Excel before write if you want to verify live update
    # time.sleep(5)
    main()
