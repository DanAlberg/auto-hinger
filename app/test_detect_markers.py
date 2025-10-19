import argparse
import glob
import json
import os
import sys
import time
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np

# Ensure we can import local helpers when run via "uv run python test_detect_markers.py"
sys.path.append(os.path.dirname(__file__))


def find_latest_png(images_dir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(images_dir, "*.png")), key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else ""


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def base_slug_for_image(img_path: str) -> str:
    name = os.path.splitext(os.path.basename(img_path))[0]
    # Prefer a friendly default name if the file has a long timestamp-like name
    return "test_image" if name.isdigit() or len(name) > 20 else name


def debug_dir_for_image(img_path: str) -> str:
    slug = base_slug_for_image(img_path)
    d = os.path.join("images", "debug", slug)
    ensure_dir(d)
    return d


def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not load image: {path}")
    return img


def load_template_with_alpha(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns (tpl_gray, alpha or None) for the template. Handles 4-channel PNGs.
    """
    tpl_rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if tpl_rgba is None:
        raise RuntimeError(f"Could not load template: {path}")
    if tpl_rgba.ndim == 3 and tpl_rgba.shape[2] == 4:
        bgr = tpl_rgba[:, :, :3]
        alpha = tpl_rgba[:, :, 3]
    else:
        if tpl_rgba.ndim == 2:  # already gray
            bgr = cv2.cvtColor(tpl_rgba, cv2.COLOR_GRAY2BGR)
        else:
            bgr = tpl_rgba  # 3-channel BGR
        alpha = None
    tpl_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return tpl_gray, alpha


def parse_xy(s: str) -> Tuple[int, int]:
    s = s.strip()
    if not s:
        raise ValueError("empty coordinate")
    if "," not in s:
        raise ValueError(f"invalid coordinate '{s}', expected 'x,y'")
    xs, ys = s.split(",", 1)
    return int(xs.strip()), int(ys.strip())


def crop_box(img: np.ndarray, tl: Tuple[int, int], br: Tuple[int, int]) -> np.ndarray:
    x1, y1 = tl
    x2, y2 = br
    x1, y1 = max(0, x1), max(0, y1)
    h, w = img.shape[:2]
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid crop box {tl}..{br}")
    return img[y1:y2, x1:x2].copy()


def save_template_from_box(
    img: np.ndarray,
    tl: Tuple[int, int],
    br: Tuple[int, int],
    out_path: str
) -> Dict[str, Any]:
    tpl = crop_box(img, tl, br)
    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, tpl)
    w = tpl.shape[1]
    h = tpl.shape[0]
    cx = tl[0] + w // 2
    cy = tl[1] + h // 2
    return {"path": out_path, "tl": tl, "br": br, "w": w, "h": h, "center": (cx, cy)}


def match_template_topk(
    img: np.ndarray,
    tpl_gray: np.ndarray,
    alpha: Optional[np.ndarray],
    roi_top: float,
    roi_bottom: float,
    scales: List[float],
    use_edges: bool,
    top_k: int,
    min_conf: float,
) -> List[Dict[str, Any]]:
    """
    Run multi-scale template matching; return top-K candidate boxes across all scales.
    Each candidate dict: {x, y, conf, width, height, top_left_x, top_left_y, scale, method}
    """
    h, w = img.shape[:2]
    y0 = max(0, int(h * roi_top))
    y1 = min(h, int(h * roi_bottom))
    roi = img[y0:y1, :]

    # Prepare ROI representations
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_edges = cv2.Canny(roi_gray, 50, 150) if use_edges else None

    candidates: List[Dict[str, Any]] = []

    for s in scales:
        tw = max(1, int(tpl_gray.shape[1] * s))
        th = max(1, int(tpl_gray.shape[0] * s))
        if tw <= 1 or th <= 1:
            continue

        tpl_s = cv2.resize(tpl_gray, (tw, th), interpolation=cv2.INTER_AREA)

        # Choose method/inputs
        if use_edges:
            if roi_edges is None or roi_edges.shape[0] < th or roi_edges.shape[1] < tw:
                continue
            tpl_proc = cv2.Canny(tpl_s, 50, 150)
            if roi_edges.shape[0] < tpl_proc.shape[0] or roi_edges.shape[1] < tpl_proc.shape[1]:
                continue
            method = cv2.TM_CCOEFF_NORMED
            res = cv2.matchTemplate(roi_edges, tpl_proc, method)
        else:
            roi_proc = roi_gray
            if roi_proc.shape[0] < th or roi_proc.shape[1] < tw:
                continue
            # If alpha provided, use CCORR with mask; else use CCORR + stroke-mask for white glyphs
            if alpha is not None:
                mask_s = cv2.resize(alpha, (tw, th), interpolation=cv2.INTER_NEAREST)
                if mask_s.dtype != np.uint8:
                    mask_s = mask_s.astype(np.uint8)
                method = cv2.TM_CCORR_NORMED
                res = cv2.matchTemplate(roi_proc, tpl_s, method, mask=mask_s)
            else:
                # stroke-only mask for high/near-white templates
                _, mask_s = cv2.threshold(tpl_s, 250, 255, cv2.THRESH_BINARY_INV)
                mask_s = mask_s.astype(np.uint8)
                if mask_s.shape != tpl_s.shape:
                    mask_s = cv2.resize(mask_s, (tw, th), interpolation=cv2.INTER_NEAREST)
                method = cv2.TM_CCORR_NORMED
                res = cv2.matchTemplate(roi_proc, tpl_s, method, mask=mask_s)

        # Extract top-K peaks for this scale by iterative NMS
        local = res.copy()
        for _ in range(top_k):
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(local)
            conf = float(max_val)
            if conf < min_conf:
                break
            px, py = int(max_loc[0]), int(max_loc[1])  # in ROI coords
            full_x = px + tw // 2
            full_y = y0 + (py + th // 2)
            candidates.append({
                'x': int(full_x),
                'y': int(full_y),
                'conf': conf,
                'width': int(tw),
                'height': int(th),
                'top_left_x': int(px),
                'top_left_y': int(y0 + py),
                'scale': float(s),
                'method': int(method),
            })
            # Suppress this neighborhood
            x1s = max(0, px - tw // 2)
            y1s = max(0, py - th // 2)
            x2s = min(local.shape[1], px + tw // 2)
            y2s = min(local.shape[0], py + th // 2)
            local[y1s:y2s, x1s:x2s] = 0.0

    # Sort overall by conf desc and return top-K overall
    candidates.sort(key=lambda c: c['conf'], reverse=True)
    return candidates[:top_k]


def draw_candidates_overlay(
    img: np.ndarray,
    age_cands: List[Dict[str, Any]],
    gen_cands: List[Dict[str, Any]],
    out_path: str
) -> None:
    dbg = img.copy()
    # Age candidates: green boxes
    for i, c in enumerate(age_cands):
        tl = (c['top_left_x'], c['top_left_y'])
        br = (tl[0] + c['width'], tl[1] + c['height'])
        cv2.rectangle(dbg, tl, br, (0, 255, 0), 2)
        cv2.putText(dbg, f"A#{i+1} conf={c['conf']:.2f} s={c['scale']:.2f}", (tl[0], max(0, tl[1]-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2, cv2.LINE_AA)
    # Gender candidates: blue boxes
    for i, c in enumerate(gen_cands):
        tl = (c['top_left_x'], c['top_left_y'])
        br = (tl[0] + c['width'], tl[1] + c['height'])
        cv2.rectangle(dbg, tl, br, (255, 0, 0), 2)
        cv2.putText(dbg, f"G#{i+1} conf={c['conf']:.2f} s={c['scale']:.2f}", (tl[0], max(0, tl[1]-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, dbg)


def draw_consensus_overlay(
    img: np.ndarray,
    age_best: Optional[Dict[str, Any]],
    gen_best: Optional[Dict[str, Any]],
    tol_px: int,
    out_path: str
) -> Dict[str, Any]:
    dbg = img.copy()
    h, w = img.shape[:2]
    y_avg = None
    delta_y = None
    error = None

    if age_best:
        y1 = int(age_best['y'])
        cv2.line(dbg, (0, y1), (w, y1), (0, 255, 0), 2)
    if gen_best:
        y2 = int(gen_best['y'])
        cv2.line(dbg, (0, y2), (w, y2), (255, 0, 0), 2)

    if age_best and gen_best:
        y1 = int(age_best['y'])
        y2 = int(gen_best['y'])
        delta_y = abs(y1 - y2)
        if delta_y <= tol_px:
            y_avg = int(round((y1 + y2) / 2))
            cv2.line(dbg, (0, y_avg), (w, y_avg), (255, 0, 255), 2)
        else:
            error = "y_mismatch"
            # translucent red band covering the gap
            band_top = min(y1, y2)
            band_bot = max(y1, y2)
            overlay = dbg.copy()
            cv2.rectangle(overlay, (0, band_top), (w, band_bot), (0, 0, 255), -1)
            alpha = 0.15
            cv2.addWeighted(overlay, alpha, dbg, 1 - alpha, 0, dbg)

    label = f"consensus_y={y_avg if y_avg is not None else 'NA'} tol={tol_px}"
    if delta_y is not None:
        label += f" delta_y={delta_y}"
    if error:
        label += f" err={error}"
    cv2.putText(dbg, label, (10, max(30, int(h*0.03))), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, dbg)
    return {"consensus_y": y_avg, "delta_y": delta_y, "error": error}


def focused_rerun(
    img: np.ndarray,
    tpl_gray: np.ndarray,
    alpha: Optional[np.ndarray],
    pivot_y: int,
    band_px: int,
    scales: List[float],
    use_edges: bool,
    top_k: int,
    min_conf: float
) -> List[Dict[str, Any]]:
    h, _ = img.shape[:2]
    roi_top = max(0.0, (pivot_y - band_px) / h)
    roi_bottom = min(1.0, (pivot_y + band_px) / h)
    return match_template_topk(
        img=img,
        tpl_gray=tpl_gray,
        alpha=alpha,
        roi_top=roi_top,
        roi_bottom=roi_bottom,
        scales=scales,
        use_edges=use_edges,
        top_k=top_k,
        min_conf=min_conf
    )


def write_json_sidecar(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Detect and annotate age/gender markers with staged overlays.")
    parser.add_argument("--image", type=str, default="", help="Path to screenshot PNG. Defaults to latest in images/ if not provided.")
    parser.add_argument("--age_tpl", type=str, default="assets/icon_age.png", help="Path to age icon template PNG.")
    parser.add_argument("--gender_tpl", type=str, default="assets/icon_gender.png", help="Path to gender icon template PNG.")
    parser.add_argument("--roi_top", type=float, default=0.0, help="Top of vertical ROI as fraction of height (default 0.0)")
    parser.add_argument("--roi_bottom", type=float, default=0.55, help="Bottom of vertical ROI as fraction of height (default 0.55)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Min confidence to accept candidates (default 0.55)")
    parser.add_argument("--no_edges", action="store_true", help="Disable edge-based matching (use grayscale/alpha mask only)")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K candidates per template to visualize (default 5)")
    parser.add_argument("--tol_px", type=int, default=12, help="Tolerance in pixels for consensus Y (default 12)")
    parser.add_argument("--step", action="store_true", help="Pause after each stage and wait for Enter")
    # Calibration options
    parser.add_argument("--calibrate", action="store_true", help="Crop templates from the provided image using given boxes.")
    parser.add_argument("--age_tl", type=str, default="", help="Age top-left as 'x,y'")
    parser.add_argument("--age_br", type=str, default="", help="Age bottom-right as 'x,y'")
    parser.add_argument("--gen_tl", type=str, default="", help="Gender top-left as 'x,y'")
    parser.add_argument("--gen_br", type=str, default="", help="Gender bottom-right as 'x,y'")
    args = parser.parse_args()

    img_path = args.image.strip()
    if not img_path:
        img_path = find_latest_png("images")
    if not img_path or not os.path.exists(img_path):
        print("❌ No screenshot found. Provide --image or ensure images/ contains PNGs.")
        sys.exit(1)

    # Friendly copy/alias: images/test_image.png
    try:
        alias = os.path.join("images", "test_image.png")
        if os.path.abspath(img_path) != os.path.abspath(alias):
            try:
                img_data = load_image_bgr(img_path)
                cv2.imwrite(alias, img_data)
                print(f"📄 Aliased baseline as {alias}")
                img_path = alias
            except Exception:
                pass
    except Exception:
        pass

    # Load image and optionally calibrate templates
    img = load_image_bgr(img_path)
    h, w = img.shape[:2]
    calibration_info: Dict[str, Any] = {}

    if args.calibrate:
        if not (args.age_tl and args.age_br and args.gen_tl and args.gen_br):
            print("❌ Calibration requires --age_tl, --age_br, --gen_tl, --gen_br (all as 'x,y').")
            sys.exit(1)
        try:
            age_tl = parse_xy(args.age_tl)
            age_br = parse_xy(args.age_br)
            gen_tl = parse_xy(args.gen_tl)
            gen_br = parse_xy(args.gen_br)
        except Exception as e:
            print(f"❌ Failed to parse calibration coords: {e}")
            sys.exit(1)

        cal_dir = os.path.join("assets", "calibrated")
        ensure_dir(cal_dir)
        age_out = os.path.join(cal_dir, "age_cal.png")
        gen_out = os.path.join(cal_dir, "gender_cal.png")

        age_meta = save_template_from_box(img, age_tl, age_br, age_out)
        gen_meta = save_template_from_box(img, gen_tl, gen_br, gen_out)

        # Override templates to calibrated ones
        args.age_tpl = age_out
        args.gender_tpl = gen_out

        # Set a focused ROI band around the average Y of the provided boxes
        pivot_y = int((age_meta["center"][1] + gen_meta["center"][1]) / 2)
        band_px = max(40, int(h * 0.05))  # ~5% of height
        args.roi_top = max(0.0, (pivot_y - band_px) / h)
        args.roi_bottom = min(1.0, (pivot_y + band_px) / h)

        calibration_info = {
            "age_box": age_meta,
            "gender_box": gen_meta,
            "pivot_y": pivot_y,
            "band_px": band_px,
            "roi_top": args.roi_top,
            "roi_bottom": args.roi_bottom
        }
        print(f"🧭 Calibration: pivot_y={pivot_y} band={band_px}px roi=({args.roi_top:.3f},{args.roi_bottom:.3f})")
        print(f"🧩 Calibrated templates saved: {age_out}, {gen_out}")

    if not os.path.exists(args.age_tpl):
        print(f"❌ Age template not found: {args.age_tpl}")
        sys.exit(1)
    if not os.path.exists(args.gender_tpl):
        print(f"❌ Gender template not found: {args.gender_tpl}")
        sys.exit(1)

    print("🧪 Marker detection on:", img_path)
    print("   Age template:   ", args.age_tpl)
    print("   Gender template:", args.gender_tpl)

    # Load templates
    try:
        age_tpl, age_alpha = load_template_with_alpha(args.age_tpl)
        gen_tpl, gen_alpha = load_template_with_alpha(args.gender_tpl)
    except Exception as e:
        print(f"❌ Template load failed: {e}")
        sys.exit(1)

    # Scales: UI icons are often smaller than template; sweep downwards too
    scales = [round(s, 2) for s in np.arange(0.5, 1.61, 0.05).tolist()]
    use_edges = not args.no_edges

    debug_dir = debug_dir_for_image(img_path)
    ts = int(time.time() * 1000)

    # Stage B: top-K candidates across ROI
    age_cands = match_template_topk(
        img, age_tpl, age_alpha,
        roi_top=args.roi_top, roi_bottom=args.roi_bottom,
        scales=scales, use_edges=use_edges,
        top_k=args.top_k, min_conf=float(args.threshold)
    )
    gen_cands = match_template_topk(
        img, gen_tpl, gen_alpha,
        roi_top=args.roi_top, roi_bottom=args.roi_bottom,
        scales=scales, use_edges=use_edges,
        top_k=args.top_k, min_conf=float(args.threshold)
    )

    out_candidates = os.path.join(
        debug_dir, f"candidates__roi-{args.roi_top:.2f}-{args.roi_bottom:.2f}__thr-{args.threshold:.2f}__{ts}.png"
    )
    draw_candidates_overlay(img, age_cands, gen_cands, out_candidates)
    print(f"🖼️  Stage B (candidates) overlay: {os.path.abspath(out_candidates)}")
    if args.step:
        input("Press Enter to continue to consensus...")

    # Stage C: consensus overlay using best from each list (if present)
    age_best = age_cands[0] if age_cands else None
    gen_best = gen_cands[0] if gen_cands else None
    out_consensus = os.path.join(debug_dir, f"consensus__tol-{args.tol_px}__{ts}.png")
    consensus_info = draw_consensus_overlay(img, age_best, gen_best, tol_px=int(args.tol_px), out_path=out_consensus)
    print(f"🖼️  Stage C (consensus) overlay: {os.path.abspath(out_consensus)}  -> {consensus_info}")
    if args.step:
        input("Press Enter to continue to focused ROI...")

    # Stage D: focused ROI rerun (only when mismatch)
    focused_outputs = None
    if age_best and gen_best and (consensus_info.get("error") == "y_mismatch"):
        # Pivot on gender (often more stable), refine age in a band around gender y
        pivot = int(gen_best["y"])
        band_px = max(40, int(h * 0.05))  # ~5% of height band
        refined_age = focused_rerun(
            img, age_tpl, age_alpha, pivot_y=pivot, band_px=band_px,
            scales=scales, use_edges=use_edges, top_k=args.top_k, min_conf=float(args.threshold)
        )
        out_refined = os.path.join(debug_dir, f"refined_age__band-{band_px}px__{ts}.png")
        draw_candidates_overlay(img, refined_age, gen_cands[:1], out_refined)
        print(f"🖼️  Stage D (focused) overlay: {os.path.abspath(out_refined)}")

        # Recompute consensus with refined best age (if better)
        age_best2 = refined_age[0] if refined_age else age_best
        out_consensus2 = os.path.join(debug_dir, f"consensus_refined__tol-{args.tol_px}__{ts}.png")
        consensus_info2 = draw_consensus_overlay(img, age_best2, gen_best, tol_px=int(args.tol_px), out_path=out_consensus2)
        print(f"🖼️  Stage D (consensus refined) overlay: {os.path.abspath(out_consensus2)}  -> {consensus_info2}")
        focused_outputs = {
            "refined_overlay": out_refined,
            "consensus_refined_overlay": out_consensus2,
            "consensus_refined": consensus_info2
        }

    # JSON sidecar
    meta = {
        "image": img_path,
        "age_template": args.age_tpl,
        "gender_template": args.gender_tpl,
        "params": {
            "roi_top": args.roi_top,
            "roi_bottom": args.roi_bottom,
            "threshold": args.threshold,
            "use_edges": use_edges,
            "top_k": args.top_k,
            "tol_px": args.tol_px,
            "scales": scales[:],  # copy
        },
        "calibration": calibration_info or None,
        "stage_outputs": {
            "candidates_overlay": out_candidates,
            "consensus_overlay": out_consensus,
            "focused": focused_outputs,
        },
        "candidates": {
            "age": age_cands,
            "gender": gen_cands
        },
        "consensus": consensus_info
    }
    sidecar = os.path.join(debug_dir, f"detect_meta__{ts}.json")
    write_json_sidecar(sidecar, meta)
    print(f"📝 JSON sidecar: {os.path.abspath(sidecar)}")

    # Exit code for CI-like usage
    success = (consensus_info.get("consensus_y") is not None)
    sys.exit(0 if success else 2)


if __name__ == "__main__":
    main()
