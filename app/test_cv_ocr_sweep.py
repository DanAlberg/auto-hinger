import argparse
import glob
import json
import os
import time
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np

# Local imports
from cv_biometrics import (
    find_biometrics_band_y,
    detect_icons_in_band,
    ocr_text_for_detection,
    parse_and_accumulate,
    _text_roi_for_icon,  # internal helper for drawing ROI
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def latest_png(images_dir: str) -> str:
    cands = sorted(glob.glob(os.path.join(images_dir, "*.png")), key=os.path.getmtime, reverse=True)
    return cands[0] if cands else ""


def draw_overlay(
    img_path: str,
    y: int,
    band_px: int,
    detections: list,
    ocr_map: Dict[int, Dict[str, Any]],
    out_path: str
) -> None:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return
    H, W = img.shape[:2]
    dbg = img.copy()

    # Draw band lines
    y0 = max(0, int(y - band_px // 2))
    y1 = min(H - 1, int(y + band_px // 2))
    cv2.line(dbg, (0, y0), (W, y0), (255, 255, 0), 1)
    cv2.line(dbg, (0, y1), (W, y1), (255, 255, 0), 1)
    cv2.line(dbg, (0, y), (W, y), (255, 0, 255), 1)

    # Draw each detection and its text ROI + OCR
    for idx, det in enumerate(detections):
        cx, cy, w, h = det.x, det.y, det.w, det.h
        tl = (int(cx - w // 2), int(cy - h // 2))
        br = (int(cx + w // 2), int(cy + h // 2))
        cv2.rectangle(dbg, tl, br, (0, 255, 0), 2)
        label = f"{det.icon} conf={det.conf:.2f}"
        cv2.putText(dbg, label, (tl[0], max(0, tl[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2, cv2.LINE_AA)

        # Text ROI box
        rx, ry, rw, rh = _text_roi_for_icon(det, W, H)
        if rw > 0 and rh > 0:
            cv2.rectangle(dbg, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
            ocr_info = ocr_map.get(idx) or {}
            ocr_text = ocr_info.get("text", "")
            if ocr_text:
                txt = (ocr_text[:36] + "…") if len(ocr_text) > 36 else ocr_text
                cv2.putText(dbg, txt, (rx, max(0, ry - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, dbg)


def run_static_extraction(
    image_path: str,
    ocr_engine: str = "easyocr",
    band_height_ratio: float = 0.06,
    allow_edges_fallback: bool = False
) -> Dict[str, Any]:
    """
    Static (non-ADB) run on a single screenshot:
    - Find biometrics row Y
    - Detect icons in band
    - OCR text to the right of each icon
    - Parse & accumulate into normalized dict
    """
    if not os.path.exists(image_path):
        raise RuntimeError(f"Image not found: {image_path}")

    # Y detection
    t0 = time.perf_counter()
    y_info = find_biometrics_band_y(
        image_path,
        templates=("assets/icon_age.png", "assets/icon_gender.png", "assets/icon_height.png"),
        roi_top=0.0,
        roi_bottom=0.55,
        threshold=0.70,
        use_edges=True,
        allow_edges_fallback=allow_edges_fallback,
        edges_roi=(0.15, 0.60),
    )
    dt_y = int((time.perf_counter() - t0) * 1000)
    if not y_info.get("found"):
        return {
            "ok": False,
            "reason": "y_not_found",
            "timing_ms": {"y_detect_ms": dt_y},
        }
    y = int(y_info["y"])

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    H, W = img.shape[:2] if img is not None else (0, 0)
    band_px = max(40, int(H * float(band_height_ratio if band_height_ratio else 0.06)))

    # Icon detection
    t1 = time.perf_counter()
    detections = detect_icons_in_band(
        image_path,
        y=y,
        band_px=band_px,
        threshold=0.55,
        use_edges=True,
        expected_px=60,
        scale_tolerance=0.30,
        min_px=20,
        max_roi_frac=0.12,
        edges_dilate_iter=1,
    )
    dt_icon = int((time.perf_counter() - t1) * 1000)

    # OCR per detection and accumulate
    acc: Dict[str, Any] = {}
    ocr_map: Dict[int, Dict[str, Any]] = {}
    t_ocr_total = 0
    t_parse_total = 0
    for idx, det in enumerate(detections):
        t_ocr = time.perf_counter()
        ocr = ocr_text_for_detection(image_path, det, engine=ocr_engine)
        t_ocr_total += int((time.perf_counter() - t_ocr) * 1000)

        t_p = time.perf_counter()
        parse_and_accumulate(det.icon, ocr, acc)
        t_parse_total += int((time.perf_counter() - t_p) * 1000)

        ocr_map[idx] = {"text": ocr.text, "conf": float(ocr.conf or 0.0), "icon": det.icon}

    # Consolidate simple biometrics view (same keys as cv_biometrics.extract_biometrics_from_carousel)
    biometrics = {
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
    }

    return {
        "ok": True,
        "biometrics": biometrics,
        "y_band": {"y": y, "band_px": band_px, "method": y_info.get("method", "two_of_three"), "found": True},
        "detections": [
            {
                "icon": d.icon,
                "x": d.x,
                "y": d.y,
                "w": d.w,
                "h": d.h,
                "conf": d.conf,
                "ocr_text": ocr_map[i]["text"],
                "ocr_conf": ocr_map[i]["conf"],
            }
            for i, d in enumerate(detections)
        ],
        "timing_ms": {
            "y_detect_ms": dt_y,
            "icon_detect_ms": dt_icon,
            "ocr_ms": t_ocr_total,
            "parse_ms": t_parse_total,
            "total_ms": dt_y + dt_icon + t_ocr_total + t_parse_total,
        },
        "ocr_map": ocr_map,
    }


def main():
    ap = argparse.ArgumentParser(description="Static CV+OCR biometrics extractor (no ADB).")
    ap.add_argument("--image", type=str, default="", help="Path to screenshot PNG (defaults to latest in images/).")
    ap.add_argument("--engine", type=str, default="easyocr", choices=["easyocr", "tesseract"], help="OCR engine.")
    ap.add_argument("--band_ratio", type=float, default=0.06, help="Vertical band height ratio around Y.")
    ap.add_argument("--edges_fallback", action="store_true", help="Allow edges-based Y fallback if templates fail.")
    ap.add_argument("--save_overlay", action="store_true", help="Save debug overlay image.")
    args = ap.parse_args()

    img_path = args.image.strip()
    if not img_path:
        img_path = latest_png("images")
    if not img_path:
        print("❌ No image found. Provide --image or place screenshots in app/images/")
        return

    print(f"🧪 Static CV+OCR on: {img_path}")
    result = run_static_extraction(
        image_path=img_path,
        ocr_engine=args.engine,
        band_height_ratio=float(args.band_ratio),
        allow_edges_fallback=bool(args.edges_fallback),
    )

    # Save JSON sidecar and optional overlay
    slug = os.path.splitext(os.path.basename(img_path))[0]
    debug_dir = os.path.join("images", "debug", slug)
    ensure_dir(debug_dir)
    sidecar = os.path.join(debug_dir, f"biometrics_static_{int(time.time()*1000)}.json")

    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"📝 JSON sidecar: {os.path.abspath(sidecar)}")

    if result.get("ok") and args.save_overlay:
        y_band = result.get("y_band", {})
        y = int(y_band.get("y", 0))
        band_px = int(y_band.get("band_px", max(40, int(0.06 * 2000))))
        # Rebuild detections to carry geometry for overlay
        dets = detect_icons_in_band(img_path, y=y, band_px=band_px)
        draw_path = os.path.join(debug_dir, f"biometrics_overlay_{int(time.time()*1000)}.png")
        # Map index to ocr text for labels
        ocr_map = {}
        for i, d in enumerate(dets):
            # naive lookup (may differ if thresholds changed between calls)
            ocr_map[i] = {"text": d.icon, "conf": float(d.conf or 0.0)}
        draw_overlay(img_path, y, band_px, dets, ocr_map, draw_path)
        print(f"🖼️ Overlay saved: {os.path.abspath(draw_path)}")

    # Print concise biometrics summary
    print("📊 Biometrics:", json.dumps(result.get("biometrics", {}), indent=2))
    print("⏱️ Timing (ms):", result.get("timing_ms", {}))


if __name__ == "__main__":
    main()
