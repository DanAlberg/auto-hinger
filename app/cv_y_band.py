"""
Y-band detection helpers for the horizontal biometrics carousel.
All other CV/OCR functionality has been deprecated and removed.
"""

import os
from typing import Any, Dict, Tuple

# Reuse robust CV utilities and ADB I/O
from helper_functions import (
    detect_age_row_dual_templates,
    infer_carousel_y_by_edges,
)


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
