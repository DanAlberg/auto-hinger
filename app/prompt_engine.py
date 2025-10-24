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
        '  "Location": "",\n'
        '  "Ethnicity": "",\n'
        '  "Children": "",  // Must be one of: "Don\'t have children", "Have children".\n'
        '  "Family plans": "",  // Must be one of: "Don\'t want children", "Want children", "Open to children", "Not sure yet".\n'
        '  "Covid Vaccine": "",  // Must be one of: "Vaccinated", "Partially vaccinated", "Not yet vaccinated".\n'
        '  "Pets": "",\n'
        '  "Zodiac Sign": "",\n'
        '  "Job title": "",\n'
        '  "University": "",\n'
        '  "Religious Beliefs": "",\n'
        '  "Home town": "",\n'
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
        "}\n\n"
        "Rules:\n"
        "- Return only the JSON object, with no commentary, markdown, or code fences.\n"
        "- Do not include any text before or after the JSON.\n"
        "- Do not use synonyms or variations for categorical values.\n"
        "- If a field is not visible, leave it empty or null.\n"
        "- For 'Profile Prompts and Answers', extract three visible prompt/answer pairs.\n"
        "- For 'Other text', include any visible text that doesn’t fit the above categories (e.g., subtext, taglines, or captions).\n"
        "- One of the images will be a stitched image containing biometrics. This is part of the profile and should be included in the results."
    )


if __name__ == "__main__":
    # For quick verification
    print(build_structured_profile_prompt())
