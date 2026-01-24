import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2

from helper_functions import (
    detect_age_row_dual_templates,
    infer_carousel_y_by_edges,
    are_images_similar_roi,
    capture_screenshot,
    swipe,
    get_screen_resolution,
)


def _timing_now_ms() -> int:
    return int(time.perf_counter() * 1000)


def find_biometrics_band_y(
    screenshot_path: str,
    *,
    templates: Tuple[str, str, str] = ("assets/icon_age.png", "assets/icon_gender.png", "assets/icon_height.png"),
    roi_top: float = 0.0,
    roi_bottom: float = 0.55,
    threshold: float = 0.70,
    use_edges: bool = True,
    tol_px: int = 12,
    tol_ratio: float = 0.01,
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


def extract_biometrics_from_carousel(
    device: Any,
    *,
    start_screenshot: Optional[str] = None,
    max_micro_swipes: int = 8,
    micro_swipe_ratio: float = 0.25,
    seek_swipe_ratio: float = 0.60,
    band_height_ratio: float = 0.10,
    allow_edges_fallback: bool = False,
    y_override: Optional[int] = None,
    band_px_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Horizontal sweep for biometrics row:
    - Find Y band (or use override)
    - Swipe horizontally to capture frames
    - Stitch Y-band slices into a single image
    Returns dict with stitched path and timing.
    """
    timings: Dict[str, int] = {"capture_ms": 0, "y_detect_ms": 0, "swipe_ms": 0}
    t_all0 = _timing_now_ms()

    W, H = get_screen_resolution(device)

    # Capture starting frame if not provided
    t0 = _timing_now_ms()
    if not start_screenshot:
        start_screenshot = capture_screenshot(device, "biometrics_start")
    timings["capture_ms"] += max(0, _timing_now_ms() - t0)

    # Detect Y band
    if y_override is not None:
        y_info = {"found": True, "y": int(y_override), "method": "override"}
    else:
        t0 = _timing_now_ms()
        y_info = find_biometrics_band_y(
            start_screenshot,
            templates=("assets/icon_age.png", "assets/icon_gender.png", "assets/icon_height.png"),
            roi_top=0.0,
            roi_bottom=0.55,
            threshold=0.70,
            use_edges=True,
            tol_px=12,
            tol_ratio=0.01,
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
            "stitched_carousel": None,
            "y_band": {"found": False},
            "timing": timings,
        }

    y = int(y_info["y"])
    band_px = int(band_px_override) if (band_px_override and band_px_override > 0) else max(60, int(H * float(band_height_ratio)))

    # Horizontal swipe params
    target_cx = int(W * 0.30)
    micro_dx = int(W * float(micro_swipe_ratio))
    seek_dx = int(W * float(seek_swipe_ratio))

    frames: List[str] = []
    last_kept_path: Optional[str] = None
    stable = 0

    left_margin = int(0.10 * W)
    right_margin = max(left_margin + 1, int(0.90 * W))

    def _hswipe(dx: int, duration_ms: int = 400) -> None:
        x1 = target_cx + dx
        x2 = target_cx - dx
        x1 = min(max(x1, left_margin), right_margin)
        x2 = min(max(x2, left_margin), right_margin)
        t_sw = _timing_now_ms()
        swipe(device, int(x1), int(y), int(x2), int(y), duration=max(350, duration_ms))
        timings["swipe_ms"] += max(0, _timing_now_ms() - t_sw)
        time.sleep(1.6)

    for step in range(max_micro_swipes):
        t0 = _timing_now_ms()
        shot = capture_screenshot(device, f"hrow_cv_{step}")
        timings["capture_ms"] += max(0, _timing_now_ms() - t0)

        if last_kept_path is not None:
            # Focus similarity on a narrow band around Y
            narrow_band_ratio = 0.10
            if are_images_similar_roi(
                shot,
                last_kept_path,
                y_center=y,
                band_ratio=narrow_band_ratio,
                hash_size=12,
                threshold=1,
            ):
                stable += 1
            else:
                stable = 0

        last_kept_path = shot
        frames.append(shot)

        if stable >= 2:
            break

        # Micro swipe each step; if we stall, use a larger seek swipe
        if stable >= 1:
            _hswipe(seek_dx, duration_ms=420)
        else:
            _hswipe(micro_dx, duration_ms=400)

    # Stitch captured bands into a single image
    bands = []
    for f in frames:
        img = cv2.imread(f)
        if img is None:
            continue
        h, w = img.shape[:2]
        y0 = max(0, y - band_px // 2)
        y1 = min(h, y + band_px // 2)
        band = img[y0:y1, :].copy()
        bands.append(band)

    stitched_path_final = None
    if bands:
        stitched = cv2.vconcat(bands)
        # Append a static strip below the row to capture any subtext area
        base_img = cv2.imread(frames[0])
        if base_img is not None:
            h, w = base_img.shape[:2]
            y0 = min(h, y + band_px // 2)
            y1 = min(h, y + int(band_px * 5.0))
            if y1 > y0:
                biometrics_strip = base_img[y0:y1, :].copy()
                stitched = cv2.vconcat([stitched, biometrics_strip])
        os.makedirs("images", exist_ok=True)
        stitched_path_final = os.path.join("images", f"stitched_carousel_full_{int(time.time()*1000)}.png")
        cv2.imwrite(stitched_path_final, stitched)

    total_ms = max(0, _timing_now_ms() - t_all0)
    timing_report = {
        "capture_ms": timings["capture_ms"],
        "y_detect_ms": timings["y_detect_ms"],
        "swipe_ms": timings["swipe_ms"],
        "total_ms": total_ms,
    }

    return {
        "biometrics": {},
        "stitched_carousel": stitched_path_final,
        "frames": frames,
        "y_band": {"y": y, "band_px": band_px, "found": True},
        "timing": timing_report,
    }


def capture_horizontal_carousel(device: Any, y_override: Optional[int] = None) -> Dict[str, Any]:
    result = extract_biometrics_from_carousel(device, y_override=y_override)
    return {
        "stitched_carousel": result.get("stitched_carousel"),
        "timing": result.get("timing", {}),
        "frames": result.get("frames", []),
        "y_band": result.get("y_band", {}),
    }
