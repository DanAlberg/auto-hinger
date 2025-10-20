import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Reuse robust CV utilities and ADB I/O
from helper_functions import (
    detect_age_row_dual_templates,
    infer_carousel_y_by_edges,
    detect_age_icon_cv_multi,
    are_images_similar_roi,
    capture_screenshot,
    swipe,
    get_screen_resolution,
)

# OCR backend: Tesseract only (no torch/easyocr)
_TESSERACT_AVAILABLE = False
try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
    _TESSERACT_AVAILABLE = True
except Exception:
    _TESSERACT_AVAILABLE = False


# Logical icon map (zodiac intentionally omitted per requirements)
ICON_MAP: Dict[str, str] = {
    "age": "assets/icon_age.png",
    "gender": "assets/icon_gender.png",
    "sexuality": "assets/icon_sexuality.png",
    "height": "assets/icon_height.png",
    "location": "assets/icon_location.png",
    "children_family": "assets/icon_children.png",
    "covid": "assets/icon_covid.png",
    "pets": "assets/icon_pets.png",
    "drinking": "assets/icon_drinking.png",
    "smoking": "assets/icon_smoking.png",
    "marijuana": "assets/icon_marijuana.png",
    "drugs": "assets/icon_drugs.png",
}

# Pets lexicon
PETS = {"dog", "cat", "bird", "fish", "reptile"}

# Zodiac words for "listed" detection only (no icon matching)
ZODIAC_WORDS = {
    "aries", "taurus", "gemini", "cancer", "leo", "virgo",
    "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"
}

# Family plans lexicon (exact)
FAMILY_PLANS = {
    "don't want children": "Don't want children",
    "do not want children": "Don't want children",
    "dont want children": "Don't want children",
    "want children": "Want children",
    "open to children": "Open to children",
    "not sure yet": "Not sure yet",
    "prefer not to say": "Prefer not to say",
}

# Covid vaccine lexicon (exact)
COVID_CHOICES = {
    "vaccinated": "Vaccinated",
    "partially vaccinated": "Partially Vaccinated",
    "not yet vaccinated": "Not yet vaccinated",
    "prefer not to say": "Prefer not to say",
}

# Lifestyle tri-state mapping
LIFESTYLE_MAP = {
    "yes": "Yes",
    "no": "No",
    "sometimes": "Sometimes",
    "socially": "Sometimes",
    "occasionally": "Sometimes",
    "rarely": "Sometimes",
    "light": "Sometimes",
    "moderate": "Sometimes",
}


@dataclass
class DetectionItem:
    icon: str
    x: int
    y: int
    w: int
    h: int
    conf: float


@dataclass
class OCRResult:
    text: str
    conf: float


def _clamp_rect(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, 0, 0
    return x0, y0, (x1 - x0), (y1 - y0)


def _maybe_set_tesseract_cmd() -> None:
    """
    If tesseract is installed in the default Windows path but not in PATH,
    set pytesseract.pytesseract.tesseract_cmd accordingly.
    """
    try:
        exe = getattr(pytesseract, "pytesseract").tesseract_cmd
        if exe and os.path.isfile(exe):
            return
    except Exception:
        pass
    candidate = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    try:
        if os.path.isfile(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
    except Exception:
        pass


def _auto_scale(gray: np.ndarray) -> np.ndarray:
    """
    Scale ROI to a suitable height for OCR: upsample small text, lightly downsample very large bands.
    """
    h, w = gray.shape[:2]
    scale = 1.0
    if h < 28:
        scale = 2.5
    elif h < 40:
        scale = 2.0
    elif h < 60:
        scale = 1.5
    elif h > 140:
        scale = 0.75
    if abs(scale - 1.0) > 1e-3:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    return gray


def _projection_trim(bin_img: np.ndarray, pad: int = 2) -> np.ndarray:
    """
    Trim empty margins to tighten ROI. Works on binary images.
    """
    if bin_img.size == 0:
        return bin_img
    rows = np.any(bin_img > 0, axis=1)
    cols = np.any(bin_img > 0, axis=0)
    if not rows.any() or not cols.any():
        return bin_img
    y0, y1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1]) - 1
    x0, x1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1]) - 1
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(bin_img.shape[0] - 1, y1 + pad)
    x1 = min(bin_img.shape[1] - 1, x1 + pad)
    return bin_img[y0:y1 + 1, x0:x1 + 1].copy()


def _binarize(gray: np.ndarray) -> np.ndarray:
    """
    Two-stage binarization: Otsu first, fallback to adaptive if blank/low-contrast.
    """
    # Otsu on slightly blurred image
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Heuristic: if too few foreground pixels, try adaptive
    fg_ratio = float(np.count_nonzero(255 - otsu)) / max(1, otsu.size)
    if fg_ratio < 0.02 or fg_ratio > 0.98:
        adap = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9
        )
        return adap
    return otsu


def _morph_cleanup(bin_img: np.ndarray) -> np.ndarray:
    """
    Light morphology to remove specks and connect thin strokes.
    """
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


def _preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """
    High-quality preprocessing for Tesseract:
    - grayscale, CLAHE, denoise,
    - auto-scale,
    - binarize (Otsu -> adaptive),
    - light morphology,
    - trim whitespace.
    Return a 3-channel image suitable for pytesseract.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    except Exception:
        pass
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
    gray = _auto_scale(gray)
    bin_img = _binarize(gray)
    bin_img = _morph_cleanup(bin_img)
    bin_img = _projection_trim(bin_img, pad=2)
    # Convert binary to 3c BGR for pytesseract
    if bin_img.ndim == 2:
        return cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    return bin_img


def _icon_whitelist_and_psm(icon: str) -> Tuple[str, str]:
    """
    Provide tesseract character whitelist and suggested PSM per icon type.
    """
    icon = (icon or "").lower()
    if icon in ("age",):
        return ("0123456789", "--oem 1 --psm 7 -l eng")
    if icon in ("height",):
        # numbers; the parser ignores any suffix; allow space just in case
        return ("0123456789 ", "--oem 1 --psm 7 -l eng")
    if icon in ("drinking", "smoking", "marijuana", "drugs"):
        # Yes/Sometimes/No (letters only)
        return ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ", "--oem 1 --psm 7 -l eng")
    if icon in ("children_family", "covid", "pets"):
        return ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '-,;:/&()[]{}0123456789", "--oem 1 --psm 6 -l eng")
    if icon in ("gender", "sexuality", "location"):
        return ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -'&/.,", "--oem 1 --psm 6 -l eng")
    # default
    return ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 -'&/.,", "--oem 1 --psm 6 -l eng")


def _ocr_tesseract(img: np.ndarray, config: Optional[str] = None) -> OCRResult:
    """
    Run Tesseract with provided config; fallback to single-line psm if empty.
    """
    try:
        pil = Image.fromarray(img)
        cfg = config or "--oem 1 --psm 6 -l eng"
        text = pytesseract.image_to_string(pil, config=cfg) or ""
        if not text.strip():
            cfg2 = cfg.replace("--psm 6", "--psm 7") if "--psm 6" in cfg else (cfg + " --psm 7")
            text = pytesseract.image_to_string(pil, config=cfg2) or ""
        conf = 0.7 if text.strip() else 0.0  # heuristic confidence
        return OCRResult(text=text.strip(), conf=conf)
    except Exception:
        return OCRResult(text="", conf=0.0)


def _ocr(img: np.ndarray, engine: str = "tesseract") -> OCRResult:
    """
    Tesseract-only OCR (engine arg kept for compatibility).
    """
    if _TESSERACT_AVAILABLE:
        return _ocr_tesseract(img)
    return OCRResult(text="", conf=0.0)


def find_biometrics_band_y(
    screenshot_path: str,
    *,
    templates: Tuple[str, str, str] = ("assets/icon_age.png", "assets/icon_gender.png", "assets/icon_height.png"),
    roi_top: float = 0.0,
    roi_bottom: float = 0.55,
    threshold: float = 0.70,
    use_edges: bool = True,
    tol_px: int = 12,
    tol_ratio: float = 0.005,
    expected_px: int = 60,
    scale_tolerance: float = 0.30,
    min_px: int = 20,
    max_roi_frac: float = 0.12,
    edges_dilate_iter: int = 1,
    allow_edges_fallback: bool = False,
    edges_roi: Tuple[float, float] = (0.15, 0.60),
) -> Dict[str, Any]:
    """
    Use two-of-three (age, gender, height) consensus to lock on Y.
    Falls back to edges-based row strength if allowed and templates fail.
    """
    # Prefer calibrated age/gender icons if available
    tpaths = list(templates)
    try:
        age_cal = "assets/calibrated/age_cal.png"
        gender_cal = "assets/calibrated/gender_cal.png"
        if os.path.exists(age_cal):
            tpaths[0] = age_cal
        if os.path.exists(gender_cal):
            tpaths[1] = gender_cal
    except Exception:
        pass

    res = detect_age_row_dual_templates(
        screenshot_path,
        template_paths=tuple(tpaths),
        roi_top=roi_top,
        roi_bottom=roi_bottom,
        threshold=threshold,
        use_edges=use_edges,
        save_debug=True,
        tolerance_px=tol_px,
        tolerance_ratio=tol_ratio,
        require_both=True,
        expected_px=expected_px,
        scale_tolerance=scale_tolerance,
        min_px=min_px,
        max_roi_frac=max_roi_frac,
        edges_dilate_iter=edges_dilate_iter,
    )
    if res.get("found"):
        return {"found": True, "y": int(res.get("y", 0)), "method": "two_of_three"}

    if allow_edges_fallback:
        edge = infer_carousel_y_by_edges(
            screenshot_path,
            roi_top=edges_roi[0],
            roi_bottom=edges_roi[1],
            smooth_kernel=21,
        )
        if edge.get("found"):
            return {"found": True, "y": int(edge["y"]), "method": "edges"}
    return {"found": False, "y": 0, "method": "not_found"}


def detect_icons_in_band(
    screenshot_path: str,
    y: int,
    band_px: int,
    *,
    threshold: float = 0.50,
    use_edges: bool = True,
    expected_px: int = 55,
    scale_tolerance: float = 0.50,
    min_px: int = 20,
    max_roi_frac: float = 0.12,
    edges_dilate_iter: int = 1,
) -> List[DetectionItem]:
    """
    Run multi-scale matching for each icon within a tight horizontal band around Y.
    Returns top-1 detection per icon type when above threshold.
    """
    img = cv2.imread(screenshot_path, cv2.IMREAD_COLOR)
    if img is None:
        return []
    H, W = img.shape[:2]
    band_half = max(8, band_px // 2)
    roi_top = max(0.0, (y - band_half) / max(1, H))
    roi_bottom = min(1.0, (y + band_half) / max(1, H))

    detections: List[DetectionItem] = []
    for icon, tpl_path in ICON_MAP.items():
        # Swap to calibrated template if present (only for known calibrated set)
        if icon in ("age", "gender"):
            cal_path = f"assets/calibrated/{icon}_cal.png"
            if os.path.exists(cal_path):
                tpl_path = cal_path
        if not os.path.exists(tpl_path):
            continue
        res = detect_age_icon_cv_multi(
            screenshot_path,
            template_path=tpl_path,
            roi_top=roi_top,
            roi_bottom=roi_bottom,
            scales=None,
            threshold=threshold,
            use_edges=use_edges,
            save_debug=False,
            label=icon,
            expected_px=expected_px,
            scale_tolerance=scale_tolerance,
            min_px=min_px,
            max_roi_frac=max_roi_frac,
            edges_dilate_iter=edges_dilate_iter,
        )
        if res.get("found"):
            detections.append(
                DetectionItem(
                    icon=icon,
                    x=int(res.get("x", 0)),
                    y=int(res.get("y", 0)),
                    w=int(res.get("width", 0)),
                    h=int(res.get("height", 0)),
                    conf=float(res.get("confidence", 0.0)),
                )
            )
    # Sort by x to preserve left->right order
    detections.sort(key=lambda d: d.x)
    return detections


def _text_roi_for_icon(det: DetectionItem, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """
    Given icon bbox, define a text ROI to its right:
    - pad_x ~ 0.02*W, pad_y ~ 0.01*H
    - width = min(0.45*W, 6*icon_w)
    - height = icon_h + 2*pad_y
    """
    pad_x = max(4, int(0.02 * img_w))
    pad_y = max(2, int(0.01 * img_h))
    roi_x = det.x + det.w // 2 + det.w // 2 + pad_x  # start at icon's right edge + pad
    roi_y = det.y - det.h // 2 - pad_y
    roi_w = min(int(0.45 * img_w), int(6 * det.w))
    roi_h = det.h + 2 * pad_y
    rx, ry, rw, rh = _clamp_rect(roi_x, roi_y, roi_w, roi_h, img_w, img_h)
    return rx, ry, rw, rh


def ocr_text_for_detection(
    screenshot_path: str,
    detection: DetectionItem,
    engine: str = "tesseract",
) -> OCRResult:
    img = cv2.imread(screenshot_path, cv2.IMREAD_COLOR)
    if img is None:
        return OCRResult(text="", conf=0.0)
    H, W = img.shape[:2]
    rx, ry, rw, rh = _text_roi_for_icon(detection, W, H)
    if rw <= 0 or rh <= 0:
        return OCRResult(text="", conf=0.0)
    roi = img[ry: ry + rh, rx: rx + rw].copy()

    # Preprocess
    proc = _preprocess_for_ocr(roi)
    # Whitelist and PSM config based on icon type
    allow, base_cfg = _icon_whitelist_and_psm(detection.icon)
    cfg = f'{base_cfg} -c tessedit_char_whitelist="{allow}"'

    # Ensure Tesseract path if needed and OCR
    _maybe_set_tesseract_cmd()
    return _ocr_tesseract(proc, config=cfg)


def _parse_age(text: str) -> Optional[int]:
    m = re.search(r"\b(1[89]|[2-9]\d)\b", text)
    if not m:
        return None
    try:
        val = int(m.group(1))
        return val if 18 <= val <= 99 else None
    except Exception:
        return None


def _parse_height_cm(text: str) -> Optional[int]:
    # Metric-only policy; we will store numeric centimeters without suffix
    m = re.search(r"\b(\d{2,3})\s*cm\b", text.lower())
    if not m:
        # Sometimes height may be recognized without 'cm'
        m2 = re.search(r"\b(\d{2,3})\b", text)
        if not m2:
            return None
        try:
            val2 = int(m2.group(1))
            if 120 <= val2 <= 230 or 100 <= val2 <= 250:
                return val2
            return None
        except Exception:
            return None
    try:
        val = int(m.group(1))
        if 120 <= val <= 230:
            return val
        if 100 <= val <= 250:
            return val
        return None
    except Exception:
        return None


def _parse_family_children(text: str) -> Tuple[Optional[str], Optional[str]]:
    s = text.lower()
    child = None
    fam = None
    if "don't have children" in s or "dont have children" in s or "do not have children" in s:
        child = "Don't have children"
    elif "have children" in s:
        child = "Have children"
    for k, v in FAMILY_PLANS.items():
        if k in s:
            fam = v
            break
    return child, fam


def _parse_covid(text: str) -> Optional[str]:
    s = text.lower()
    for k, v in COVID_CHOICES.items():
        if k in s:
            return v
    return None


def _parse_pets(text: str) -> List[str]:
    s = text.lower()
    found = []
    for p in PETS:
        # match singular/plural simple
        if re.search(rf"\b{p}s?\b", s):
            found.append(p.capitalize())
    # Dedup and stable order by PETS declaration
    out = [p.capitalize() for p in PETS if p.capitalize() in found]
    return out


def _parse_tri_state(text: str) -> Optional[str]:
    s = text.lower()
    for k, v in LIFESTYLE_MAP.items():
        if re.search(rf"\b{k}\b", s):
            return v
    return None


def _zodiac_listed(text: str) -> bool:
    s = text.lower()
    for z in ZODIAC_WORDS:
        if re.search(rf"\b{z}\b", s):
            return True
    return False


def parse_and_accumulate(
    icon: str,
    ocr: OCRResult,
    acc: Dict[str, Any],
) -> None:
    """
    Update accumulator based on icon type and OCR text.
    - Age and height mandatory; location mandatory (string).
    - Children/family share same icon; disambiguate via text.
    """
    text = ocr.text.strip()
    conf = float(ocr.conf or 0.0)
    if not text:
        return

    # Initialize slots in acc with structure to store best candidate
    def _ensure(field: str):
        if field not in acc:
            acc[field] = {"best_text": "", "norm_value": None, "ocr_conf": 0.0, "frames_seen": 0}

    def _maybe_set(field: str, norm_value: Any):
        _ensure(field)
        # Prefer better OCR conf or if current norm_value is None
        cur = acc[field]
        if cur["norm_value"] is None or conf > float(cur.get("ocr_conf", 0.0)) + 0.05:
            cur["best_text"] = text
            cur["norm_value"] = norm_value
            cur["ocr_conf"] = conf
        cur["frames_seen"] = int(cur.get("frames_seen", 0)) + 1

    if icon == "age":
        val = _parse_age(text)
        if val is not None:
            _maybe_set("age", int(val))
        return

    if icon == "gender":
        # Keep original casing; sanitize to title case
        _maybe_set("gender", text.strip().title())
        return

    if icon == "sexuality":
        _maybe_set("sexuality", text)
        return

    if icon == "height":
        cm = _parse_height_cm(text)
        if cm is not None:
            _maybe_set("height_cm", int(cm))
        return

    if icon == "location":
        # Accept as-is
        _maybe_set("location", text)
        return

    if icon == "children_family":
        child, fam = _parse_family_children(text)
        if child:
            _maybe_set("children", child)
        if fam:
            _maybe_set("family_plans", fam)
        return

    if icon == "covid":
        v = _parse_covid(text)
        if v:
            _maybe_set("covid_vaccine", v)
        return

    if icon == "pets":
        lst = _parse_pets(text)
        if lst:
            _maybe_set("pets", lst)
        return

    if icon in ("drinking", "smoking", "marijuana", "drugs"):
        tri = _parse_tri_state(text)
        if tri:
            _maybe_set(icon, tri)
        return

    # Fallback zodiac "listed" detection: if OCR contains any zodiac words near the icon's ROI
    if icon == "zodiac":
        _maybe_set("zodiac_listed", _zodiac_listed(text))


def _timing_now_ms() -> int:
    return int(time.perf_counter() * 1000)


def _ocr_fallback_parse_band(
    screenshot_path: str,
    y: int,
    band_px: int,
    acc: Dict[str, Any],
) -> bool:
    """
    OCR the entire horizontal band as a fallback when icon detection fails.
    Extract height (cm), drinking tri-state, and infer location from remaining text.
    Returns True if any field was updated.
    """
    img = cv2.imread(screenshot_path, cv2.IMREAD_COLOR)
    if img is None:
        return False
    H, W = img.shape[:2]
    band_half = max(8, band_px // 2)
    y0 = max(0, y - band_half)
    y1 = min(H, y + band_half)
    roi = img[y0:y1, :].copy()

    proc = _preprocess_for_ocr(roi)
    _maybe_set_tesseract_cmd()
    # Broad config to capture words and numbers
    cfg = "--oem 1 --psm 6 -l eng -c preserve_interword_spaces=1"
    ocr = _ocr_tesseract(proc, config=cfg)
    text = (ocr.text or "").strip()
    if not text:
        return False

    updated = False

    # Height
    cm = _parse_height_cm(text)
    if cm is not None:
        # Insert with moderate confidence since it's band-wide
        node = acc.get("height_cm") or {"best_text": "", "norm_value": None, "ocr_conf": 0.0, "frames_seen": 0}
        if node["norm_value"] is None or ocr.conf > node["ocr_conf"] + 0.05:
            node.update({"best_text": f"{cm} cm", "norm_value": int(cm), "ocr_conf": max(node["ocr_conf"], 0.6)})
            node["frames_seen"] = int(node.get("frames_seen", 0)) + 1
            acc["height_cm"] = node
            updated = True
        # remove the height substring from text for location inference
        text = re.sub(r"\b\d{2,3}\s*cm\b", " ", text, flags=re.I)

    # Drinking tri-state
    tri = _parse_tri_state(text)
    if tri:
        node = acc.get("drinking") or {"best_text": "", "norm_value": None, "ocr_conf": 0.0, "frames_seen": 0}
        if node["norm_value"] is None or ocr.conf > node["ocr_conf"] + 0.05:
            node.update({"best_text": tri, "norm_value": tri, "ocr_conf": max(node["ocr_conf"], 0.6)})
            node["frames_seen"] = int(node.get("frames_seen", 0)) + 1
            acc["drinking"] = node
            updated = True
        # strip the matched keyword to help find location
        text = re.sub(r"\b(yes|no|sometimes|socially|occasionally|rarely|light|moderate)\b", " ", text, flags=re.I)

    # Location: pick the longest remaining alpha phrase
    # Remove stray numbers, punctuation commonly from units
    cleaned = re.sub(r"[^A-Za-z\s\-'/,]", " ", text).strip()
    # Collapse whitespace
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    # Heuristic: choose the longest token group with >= 3 letters total
    candidates = sorted(set([s.strip() for s in cleaned.split(",") if len(re.sub(r"[^A-Za-z]", "", s)) >= 3]), key=len, reverse=True)
    loc = candidates[0] if candidates else ""
    if loc:
        node = acc.get("location") or {"best_text": "", "norm_value": None, "ocr_conf": 0.0, "frames_seen": 0}
        if node["norm_value"] is None or ocr.conf > node["ocr_conf"] + 0.05:
            node.update({"best_text": loc, "norm_value": loc, "ocr_conf": max(node["ocr_conf"], 0.55)})
            node["frames_seen"] = int(node.get("frames_seen", 0)) + 1
            acc["location"] = node
            updated = True

    return updated


def extract_biometrics_from_carousel(
    device: Any,
    *,
    start_screenshot: Optional[str] = None,
    ocr_engine: str = "tesseract",
    max_micro_swipes: int = 12,
    micro_swipe_ratio: float = 0.25,
    seek_swipe_ratio: float = 0.60,
    target_center_x_ratio: float = 0.38,
    band_height_ratio: float = 0.10,
    y_method: str = "two_of_three",
    allow_edges_fallback: bool = False,
    verbose_timing: bool = False,
    y_override: Optional[int] = None,
    band_px_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Closed-loop horizontal sweep:
    - Find Y band for biometrics row
    - Iteratively center next unseen icon and OCR the text to its right
    - Stop when no new icons appear after a seek swipe or max swipes
    Returns a result dict with biometrics, frames, band info, and timing breakdown.
    """
    timings: Dict[str, int] = {
        "capture_ms": 0,
        "y_detect_ms": 0,
        "icon_detect_ms": 0,
        "ocr_ms": 0,
        "swipe_ms": 0,
        "parse_ms": 0,
    }
    t_all0 = _timing_now_ms()

    # Screen size
    W, H = get_screen_resolution(device)

    # Step 1: capture starting frame if not provided
    t0 = _timing_now_ms()
    if not start_screenshot:
        start_screenshot = capture_screenshot(device, "biometrics_start")
    timings["capture_ms"] += max(0, _timing_now_ms() - t0)

    # Step 2: lock Y band (use override if provided)
    if y_override is not None:
        y_info = {"found": True, "y": int(y_override), "method": "override"}
        timings["y_detect_ms"] += 0
    else:
        t0 = _timing_now_ms()
        y_info = find_biometrics_band_y(
            start_screenshot,
            templates=(ICON_MAP["age"], ICON_MAP["gender"], ICON_MAP["height"]),
            roi_top=0.0,
            roi_bottom=0.55,
            threshold=0.70,
            use_edges=True,
            tol_px=12,
            tol_ratio=0.005,
            expected_px=60,
            scale_tolerance=0.30,
            min_px=20,
            max_roi_frac=0.12,
            edges_dilate_iter=1,
            allow_edges_fallback=allow_edges_fallback,
            edges_roi=(0.15, 0.60),
        )
        timings["y_detect_ms"] += max(0, _timing_now_ms() - t0)

        if not y_info.get("found"):
            return {
                "biometrics": {},
                "frames": [],
                "y_band": {"found": False},
                "timing": timings,
                "debug_overlays": [],
            }

    y = int(y_info["y"])
    band_px = int(band_px_override) if (band_px_override and band_px_override > 0) else max(60, int(H * float(band_height_ratio if band_height_ratio else 0.10)))

    # Controller params
    target_cx = int(W * float(target_center_x_ratio if target_center_x_ratio else 0.38))
    micro_dx = int(W * float(micro_swipe_ratio if micro_swipe_ratio else 0.25))
    seek_dx = int(W * float(seek_swipe_ratio if seek_swipe_ratio else 0.60))

    processed_right_x = 0
    consecutive_no_new = 0
    frames: List[Dict[str, Any]] = []
    debug_paths: List[str] = []

    last_kept_path: Optional[str] = None
    seen_types: set = set()
    acc: Dict[str, Any] = {}

    # Gesture-safe horizontal corridor (avoid Pixel edge-back gesture)
    left_margin = int(0.10 * W)
    right_margin = max(left_margin + 1, int(0.90 * W))

    # Helper to perform swipe and wait a bit
    def _hswipe(dx: int, duration_ms: int = 400) -> None:
        # Compute path within safe corridor
        x1 = target_cx + dx
        x2 = target_cx - dx
        x1 = min(max(x1, left_margin), right_margin)
        x2 = min(max(x2, left_margin), right_margin)
        t_sw = _timing_now_ms()
        swipe(device, int(x1), int(y), int(x2), int(y), duration=max(350, duration_ms))
        timings["swipe_ms"] += max(0, _timing_now_ms() - t_sw)
        time.sleep(1.6)

    # Iterative closed-loop sweep
    for step in range(max_micro_swipes):
        # Capture
        t0 = _timing_now_ms()
        shot = capture_screenshot(device, f"hrow_cv_{step}")
        timings["capture_ms"] += max(0, _timing_now_ms() - t0)

        # Dedup ROI against last_kept
        if last_kept_path is not None:
            if are_images_similar_roi(
                shot,
                last_kept_path,
                y_center=y,
                band_ratio=float(band_px / max(1, H)),
                hash_size=8,
                threshold=3,  # tight for horizontal strip
            ):
                consecutive_no_new += 1
            else:
                consecutive_no_new = 0

        last_kept_path = shot

        # Detect icons in band
        t_id = _timing_now_ms()
        dets = detect_icons_in_band(
            shot,
            y=y,
            band_px=band_px,
            threshold=0.50,
            use_edges=True,
            expected_px=55,
            scale_tolerance=0.50,
            min_px=20,
            max_roi_frac=0.12,
            edges_dilate_iter=1,
        )
        timings["icon_detect_ms"] += max(0, _timing_now_ms() - t_id)

        # Choose next candidate: first with bbox.left > processed_right_x + margin
        margin = max(6, int(0.01 * W))
        next_det: Optional[DetectionItem] = None
        for d in dets:
            left = d.x - d.w // 2
            if left > processed_right_x + margin:
                next_det = d
                break

        # If no icon candidate, try OCR fallback on the band before swiping
        if next_det is None:
            updated = _ocr_fallback_parse_band(shot, y, band_px, acc)
            if updated and acc.get("height_cm") and acc.get("location"):
                # If mandatory fields gathered, we can stop early
                break
            # No visible next or incomplete: perform micro/seek swipe
            if consecutive_no_new >= 2:
                _hswipe(seek_dx, duration_ms=450)
            else:
                _hswipe(micro_dx, duration_ms=400)
            continue

        # If candidate is far from target center, nudge to center and re-capture
        delta = next_det.x - target_cx
        if abs(delta) > max(8, next_det.w):  # allow if roughly centered
            # Nudge: smaller swipe proportional to distance (remain inside safe corridor)
            nudge = int(0.7 * delta)
            _hswipe(nudge, duration_ms=360)
            # Next loop will capture and re-detect
            continue

        # OCR and parse for the centered detection
        t_ocr = _timing_now_ms()
        ocr = ocr_text_for_detection(shot, next_det, engine=ocr_engine)
        timings["ocr_ms"] += max(0, _timing_now_ms() - t_ocr)

        t_parse = _timing_now_ms()
        parse_and_accumulate(next_det.icon, ocr, acc)
        timings["parse_ms"] += max(0, _timing_now_ms() - t_parse)

        # Update processed_right_x to the right edge of this text ROI
        # Approximate: right edge at icon right + pad + 6*icon_w
        processed_right_x = min(W - 1, (next_det.x + next_det.w // 2) + max(6 * next_det.w, int(0.45 * W)))

        seen_types.add(next_det.icon)
        frames.append(
            {
                "screenshot": shot,
                "icon": next_det.icon,
                "icon_conf": next_det.conf,
                "ocr_text": ocr.text,
                "ocr_conf": ocr.conf,
                "x": next_det.x,
                "y": next_det.y,
                "w": next_det.w,
                "h": next_det.h,
            }
        )

        # Heuristic stopping: if content stabilizes and we've likely seen the row
        if consecutive_no_new >= 2 and (acc.get("height_cm") or acc.get("location")):
            break

        # Nudge to bring next item
        _hswipe(micro_dx, duration_ms=400)

    # Consolidate output biometrics
    biometrics: Dict[str, Any] = {
        "age": acc.get("age", {}).get("norm_value"),
        "gender": acc.get("gender", {}).get("norm_value"),
        "sexuality": acc.get("sexuality", {}).get("norm_value"),
        "height_cm": acc.get("height_cm", {}).get("norm_value"),
        "location": acc.get("location", {}).get("norm_value"),
        "children": acc.get("children", {}).get("norm_value"),
        "family_plans": acc.get("family_plans", {}).get("norm_value"),
        "covid_vaccine": acc.get("covid_vaccine", {}).get("norm_value"),
        "pets": acc.get("pets", {}).get("norm_value"),
        "drinking": acc.get("drinking", {}).get("norm_value"),
        "smoking": acc.get("smoking", {}).get("norm_value"),
        "marijuana": acc.get("marijuana", {}).get("norm_value"),
        "drugs": acc.get("drugs", {}).get("norm_value"),
        # boolean only; we didn't explicitly detect zodiac icon
        "zodiac_listed": None,
    }

    total_ms = max(0, _timing_now_ms() - t_all0)
    timing_report = {
        **timings,
        "total_ms": total_ms,
    }

    if verbose_timing:
        try:
            print(
                f"[CV_OCR_TIMING] total={timing_report['total_ms']}ms "
                f"capture={timing_report['capture_ms']}ms y_detect={timing_report['y_detect_ms']}ms "
                f"icon_detect={timing_report['icon_detect_ms']}ms ocr={timing_report['ocr_ms']}ms "
                f"parse={timing_report['parse_ms']}ms swipe={timing_report['swipe_ms']}ms"
            )
        except Exception:
            pass

    return {
        "biometrics": biometrics,
        "frames": frames,
        "y_band": {"y": y, "band_px": band_px, "method": y_info.get("method", y_method), "found": True},
        "timing": timing_report,
        "debug_overlays": [],
    }


def merge_biometrics_into_extracted_profile(
    extracted: Dict[str, Any],
    biometrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge CV biometrics into existing extracted_profile:
    - Fill missing keys (None) from biometrics
    - Height stored as numeric centimeters (no suffix)
    """
    out = dict(extracted or {})
    # Basic identity fields
    if out.get("age") is None and biometrics.get("age") is not None:
        out["age"] = int(biometrics["age"])
    # Height: store as numeric centimeters (no unit suffix)
    h_cm = biometrics.get("height_cm")
    if out.get("height") in (None, "", 0) and isinstance(h_cm, int):
        out["height"] = int(h_cm)
    # Location mandatory
    if (not out.get("location")) and biometrics.get("location"):
        out["location"] = biometrics["location"]

    # Optional fields
    for k_src, k_dst in [
        ("gender", "gender"),
        ("sexuality", "sexuality"),
        ("children", "current_children"),
        ("family_plans", "family_plans"),
        ("covid_vaccine", "covid_vaccine"),
        ("drinking", "drinking"),
        ("smoking", "smoking"),
        ("marijuana", "marijuana"),
        ("drugs", "drugs"),
    ]:
        if out.get(k_dst) in (None, "", []):
            v = biometrics.get(k_src)
            if v is not None:
                out[k_dst] = v

    # Pets to boolean dict structure if possible
    pets_list = biometrics.get("pets") or []
    if isinstance(pets_list, list):
        pets_dict = out.get("pets", {}) or {}
        for p in PETS:
            key = p.lower()
            val = None
            if p.capitalize() in pets_list:
                val = True
            pets_dict[key] = val if val is not None else pets_dict.get(key)
        out["pets"] = pets_dict

    return out
