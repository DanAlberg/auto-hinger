from typing import Any, Dict, List

from text_utils import normalize_dashes

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


