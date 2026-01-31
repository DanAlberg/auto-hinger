#!/usr/bin/env python3

"""Entry point for the full Hinge scrape/score/opener pipeline."""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict

import config  # ensure .env is loaded early

from helper_functions import ensure_adb_running, connect_device, get_screen_resolution, open_hinge
from extraction import run_llm1_visual, run_profile_eval_llm, _build_extracted_profile
from openers import run_llm3_long, run_llm3_short, run_llm4
from profile_utils import _get_core, _norm_value
from runtime import _is_run_json_enabled, _log
from scoring import _classify_preference_flag, _format_score_table, _score_profile_long, _score_profile_short
from sqlite_store import (
    upsert_profile_flat,
    update_profile_opening_messages_json,
    update_profile_opening_pick,
    update_profile_verdict,
)
from ui_scan import (
    _bounds_center,
    _bounds_close,
    _clear_crops_folder,
    _compute_desired_offset,
    _dump_ui_xml,
    _ensure_photo_square,
    _find_like_button_in_photo,
    _find_like_button_near_bounds_screen,
    _find_like_button_near_expected,
    _find_poll_option_bounds_by_text,
    _find_prompt_bounds_by_text,
    _find_scroll_area,
    _find_visible_photo_bounds,
    _is_square_bounds,
    _match_photo_bounds_by_hash,
    _parse_ui_nodes,
    _resolve_target_from_ui_map,
    _scan_profile_single_pass,
    _seek_photo_by_index,
    _seek_photo_by_index_from_bottom,
    _seek_target_on_screen,
)


def _force_gemini_env() -> None:
    os.environ.setdefault("LLM_PROVIDER", "gemini")
    gemini_model = os.getenv("GEMINI_MODEL")
    gemini_small = os.getenv("GEMINI_SMALL_MODEL")
    if gemini_model and not os.getenv("LLM_MODEL"):
        os.environ["LLM_MODEL"] = gemini_model
    if gemini_small and not os.getenv("LLM_SMALL_MODEL"):
        os.environ["LLM_SMALL_MODEL"] = gemini_small
    os.environ.setdefault("HINGE_CV_DEBUG_MODE", "0")
    os.environ.setdefault("HINGE_TARGET_DEBUG", "1")
    os.environ.setdefault("HINGE_SHOW_EXTRACTION_WARNINGS", "0")


def _init_device(device_ip: str):
    ensure_adb_running()
    device = connect_device(device_ip)
    if not device:
        print("Failed to connect to device")
        return None, 0, 0
    width, height = get_screen_resolution(device)
    open_hinge(device)
    time.sleep(5)
    return device, width, height


def main() -> int:
    _force_gemini_env()

    device_ip = "127.0.0.1"
    max_scrolls = 40
    scroll_step = 700

    device, width, height = _init_device(device_ip)
    if not device or not width or not height:
        print("Device/size missing; cannot proceed.")
        return 1

    _clear_crops_folder()
    _log("[UI] Single-pass scan (slow scroll, capture as you go)...")
    scan_result = _scan_profile_single_pass(
        device,
        width,
        height,
        max_scrolls=max_scrolls,
        scroll_step_px=scroll_step,
    )
    ui_map = scan_result.get("ui_map", {})
    biometrics = scan_result.get("biometrics", {})
    photo_paths = scan_result.get("photo_paths", [])
    scroll_offset = int(scan_result.get("scroll_offset", 0))
    scroll_area = scan_result.get("scroll_area")
    scan_nodes = scan_result.get("nodes")

    _log(f"[LLM1] Sending {len(photo_paths)} photos for visual analysis")
    llm1_result, llm1_meta = run_llm1_visual(
        photo_paths,
        model=os.getenv("LLM_SMALL_MODEL") or os.getenv("GEMINI_SMALL_MODEL") or None,
    )
    extracted = _build_extracted_profile(biometrics, ui_map, llm1_result)

    eval_result = run_profile_eval_llm(
        extracted,
        model=os.getenv("LLM_SMALL_MODEL") or os.getenv("GEMINI_SMALL_MODEL") or None,
    )
    long_score_result = _score_profile_long(extracted, eval_result)
    short_score_result = _score_profile_short(extracted, eval_result)
    score_table_long = _format_score_table("Long", long_score_result)
    score_table_short = _format_score_table("Short", short_score_result)
    score_table = score_table_long + "\n\n" + score_table_short
    long_score = long_score_result.get("score", 0) if isinstance(long_score_result, dict) else 0
    short_score = short_score_result.get("score", 0) if isinstance(short_score_result, dict) else 0

    T_LONG = 15
    T_SHORT = 20
    DOM_MARGIN = 10

    long_ok = long_score >= T_LONG
    short_ok = short_score >= T_SHORT
    long_delta = long_score - T_LONG
    short_delta = short_score - T_SHORT

    if not long_ok and not short_ok:
        decision = "reject"
    elif long_ok and (not short_ok or long_delta >= short_delta + DOM_MARGIN):
        decision = "long_pickup"
    elif short_ok and (not long_ok or short_delta >= long_delta + DOM_MARGIN):
        decision = "short_pickup"
    else:
        decision = "long_pickup"

    dating_intention = _norm_value((_get_core(extracted) or {}).get("Dating Intentions", ""))
    if dating_intention in {_norm_value("Short-term relationship")}:
        if decision == "long_pickup":
            decision = "reject"
    elif dating_intention == _norm_value("Life partner"):
        if decision == "short_pickup":
            decision = "reject"

    manual_override = ""
    try:
        print(
            "Gate decision pre-override: {decision} (long_score={long_score}, short_score={short_score}, "
            "long_delta={long_delta}, short_delta={short_delta})".format(
                decision=decision,
                long_score=long_score,
                short_score=short_score,
                long_delta=long_score - T_LONG,
                short_delta=short_score - T_SHORT,
            )
        )
        override = input("Override decision? (long/short/reject, blank to keep): ").strip().lower()
        if override in {"long", "short", "reject"}:
            manual_override = override
            decision = {"long": "long_pickup", "short": "short_pickup", "reject": "reject"}[override]
    except Exception:
        pass
    print(
        "GATE decision={decision} long_score={long_score} short_score={short_score} "
        "long_delta={long_delta} short_delta={short_delta} dom_margin={dom_margin}".format(
            decision=decision,
            long_score=long_score,
            short_score=short_score,
            long_delta=long_score - T_LONG,
            short_delta=short_score - T_SHORT,
            dom_margin=DOM_MARGIN,
        )
    )

    llm3_variant = ""
    llm3_result = {}
    llm4_result = {}
    target_action = {}
    if decision == "short_pickup":
        llm3_variant = "short"
        llm3_result = run_llm3_short(extracted)
    elif decision == "long_pickup":
        llm3_variant = "long"
        llm3_result = run_llm3_long(extracted)
    if llm3_result:
        llm4_result = run_llm4(llm3_result)
        target_id = str(llm4_result.get("main_target_id", "") or "").strip()
        if target_id:
            print(f"[TARGET] LLM4 chose target_id={target_id}")
            target_info = _resolve_target_from_ui_map(ui_map, target_id)
            target_action = {"target_id": target_id, **target_info}
            target_type = target_info.get("type", "")
            if target_type == "photo":
                target_hash = target_info.get("photo_hash")
                target_index = None
                try:
                    target_index = int(str(target_id).split("_", 1)[1])
                except Exception:
                    target_index = None
                total_photos = len(ui_map.get("photos", []))
                if target_hash is None or not scroll_area:
                    print("[TARGET] missing photo hash or scroll area; skipping tap")
                elif not target_index:
                    print("[TARGET] missing photo index; skipping tap")
                else:
                    seek_photo = _seek_photo_by_index_from_bottom(
                        device,
                        width,
                        height,
                        scroll_area,
                        scan_nodes,
                        scroll_offset,
                        int(target_index),
                        total_photos,
                        target_hash=int(target_hash),
                    )
                    cur_nodes = seek_photo.get("nodes")
                    cur_scroll_area = seek_photo.get("scroll_area") or scroll_area
                    tap_bounds = seek_photo.get("tap_bounds")
                    tap_desc = seek_photo.get("tap_desc", "Like photo")
                    if not tap_bounds:
                        print("[TARGET] reverse seek failed; falling back to top-down scan")
                        seek_photo = _seek_photo_by_index(
                            device,
                            width,
                            height,
                            scroll_area,
                            int(target_index),
                            target_hash=int(target_hash),
                        )
                        cur_nodes = seek_photo.get("nodes")
                        cur_scroll_area = seek_photo.get("scroll_area") or scroll_area
                        tap_bounds = seek_photo.get("tap_bounds")
                        tap_desc = seek_photo.get("tap_desc", "Like photo")
                    if tap_bounds:
                        tap_x, tap_y = _bounds_center(tap_bounds)
                        tap_x = max(0, min(width - 1, tap_x))
                        tap_y = max(0, min(height - 1, tap_y))
                        print(f"[TARGET] photo tap bounds={tap_bounds} desc='{tap_desc}'")
                        try:
                            from helper_functions import tap
                            tap(device, tap_x, tap_y)
                            print(f"[TARGET] tap issued at ({tap_x}, {tap_y})")
                        except Exception as e:
                            print(f"[TARGET] tap failed: {e}")
                        target_action["tap_coords"] = [tap_x, tap_y]
                        target_action["tap_like"] = True
                        time.sleep(0.35)
                        post_xml = _dump_ui_xml(device)
                        post_nodes = _parse_ui_nodes(post_xml)
                        post_bounds, _ = _find_like_button_near_expected(
                            post_nodes, cur_scroll_area, "photo", tap_y
                        )
                        if _bounds_close(post_bounds, tap_bounds):
                            print("[TARGET] like button still present near tap (not confirmed)")
                        else:
                            print("[TARGET] like button not found near tap (likely tapped)")
                    else:
                        print("[TARGET] photo not found on-screen; skipping tap")
            elif target_info.get("abs_bounds") and scroll_area:
                target_bounds = target_info["abs_bounds"]
                focus_bounds = target_bounds
                if target_type == "photo" and target_info.get("photo_bounds"):
                    focus_bounds = target_info.get("photo_bounds")
                if target_type == "prompt" and target_info.get("prompt_bounds"):
                    focus_bounds = target_info.get("prompt_bounds")
                desired_offset = _compute_desired_offset(focus_bounds, scroll_area)
                seek = _seek_target_on_screen(
                    device,
                    width,
                    height,
                    scroll_area,
                    scroll_offset,
                    target_type,
                    target_info,
                    desired_offset,
                )
                scroll_offset = seek.get("scroll_offset", scroll_offset)
                cur_nodes = seek.get("nodes") or _parse_ui_nodes(_dump_ui_xml(device))
                cur_scroll_area = seek.get("scroll_area") or _find_scroll_area(cur_nodes) or scroll_area
                expected_screen_y = int((target_bounds[1] + target_bounds[3]) / 2 - scroll_offset)
                tap_bounds = None
                tap_desc = ""
                if target_type == "prompt":
                    prompt_bounds = seek.get("prompt_bounds")
                    if not prompt_bounds:
                        prompt_bounds = _find_prompt_bounds_by_text(
                            cur_nodes,
                            target_info.get("prompt", ""),
                            target_info.get("answer", ""),
                        )
                    if prompt_bounds:
                        tap_bounds, tap_desc = _find_like_button_near_bounds_screen(
                            cur_nodes, prompt_bounds, "prompt"
                        )
                        print(f"[TARGET] prompt found on-screen at {prompt_bounds}")
                    else:
                        print("[TARGET] prompt not found on-screen; falling back to expected Y")
                        tap_bounds, tap_desc = _find_like_button_near_expected(
                            cur_nodes, cur_scroll_area, "prompt", expected_screen_y
                        )
                elif target_type == "poll":
                    option_bounds = seek.get("poll_bounds")
                    if not option_bounds:
                        option_bounds = _find_poll_option_bounds_by_text(
                            cur_nodes, target_info.get("option_text", "")
                        )
                    if option_bounds:
                        tap_bounds = option_bounds
                        tap_desc = "poll_option"
                        print(f"[TARGET] poll option found on-screen at {option_bounds}")
                    else:
                        print("[TARGET] poll option not found on-screen; skipping tap")
                elif target_type == "photo":
                    target_hash = target_info.get("photo_hash")
                    target_photo_bounds = target_info.get("photo_bounds")
                    target_abs_center_y = None
                    if target_photo_bounds:
                        target_abs_center_y = int((target_photo_bounds[1] + target_photo_bounds[3]) / 2)
                        expected_screen_y = int(target_abs_center_y - scroll_offset)

                    photo_bounds = seek.get("photo_bounds")
                    if not photo_bounds and target_abs_center_y is not None:
                        photo_bounds = _find_visible_photo_bounds(
                            cur_nodes, cur_scroll_area, expected_screen_y
                        )
                    if photo_bounds:
                        cur_nodes, scroll_offset, photo_bounds = _ensure_photo_square(
                            device,
                            width,
                            height,
                            cur_scroll_area,
                            cur_nodes,
                            scroll_offset,
                            photo_bounds,
                            target_abs_center_y=target_abs_center_y,
                        )
                        cur_scroll_area = _find_scroll_area(cur_nodes) or cur_scroll_area
                        if target_abs_center_y is not None:
                            expected_screen_y = int(target_abs_center_y - scroll_offset)

                    if target_hash is None:
                        print("[TARGET] missing photo hash; skipping hash match")
                    match_bounds = seek.get("photo_match_bounds")
                    dist = None
                    if target_hash is not None and not match_bounds:
                        match_bounds, dist = _match_photo_bounds_by_hash(
                            device,
                            width,
                            height,
                            cur_nodes,
                            cur_scroll_area,
                            int(target_hash),
                            expected_screen_y=expected_screen_y,
                            max_dist=18,
                            square_only=True,
                        )
                        if match_bounds:
                            tap_bounds, tap_desc = _find_like_button_in_photo(
                                cur_nodes, match_bounds
                            )
                            print(
                                f"[TARGET] photo hash matched bounds={match_bounds} dist={dist}"
                            )

                    if not tap_bounds and photo_bounds:
                        dy = None
                        if expected_screen_y is not None:
                            dy = abs(_bounds_center(photo_bounds)[1] - expected_screen_y)
                        if _is_square_bounds(photo_bounds) and (dy is None or dy <= 220):
                            tap_bounds, tap_desc = _find_like_button_in_photo(
                                cur_nodes, photo_bounds
                            )
                            print(f"[TARGET] using closest square photo by y dist={dy}")
                    if not tap_bounds:
                        print("[TARGET] photo not found on-screen; skipping tap")
                else:
                    tap_bounds, tap_desc = _find_like_button_near_expected(
                        cur_nodes, cur_scroll_area, target_type, expected_screen_y
                    )
                if not tap_bounds:
                    if target_type in {"photo", "poll"}:
                        print(f"[TARGET] no bounds resolved for {target_type}; skipping tap")
                    else:
                        tap_bounds = (
                            target_bounds[0],
                            target_bounds[1] - scroll_offset,
                            target_bounds[2],
                            target_bounds[3] - scroll_offset,
                        )
                if tap_bounds:
                    tap_x, tap_y = _bounds_center(tap_bounds)
                    tap_x = max(0, min(width - 1, tap_x))
                    tap_y = max(0, min(height - 1, tap_y))
                    print(f"[TARGET] tap bounds={tap_bounds} desc='{tap_desc}' expected_y={expected_screen_y}")
                    try:
                        from helper_functions import tap
                        tap(device, tap_x, tap_y)
                        print(f"[TARGET] tap issued at ({tap_x}, {tap_y})")
                    except Exception as e:
                        print(f"[TARGET] tap failed: {e}")
                    target_action["tap_coords"] = [tap_x, tap_y]
                    target_action["tap_like"] = True
                    if target_type != "poll":
                        time.sleep(0.35)
                        post_xml = _dump_ui_xml(device)
                        post_nodes = _parse_ui_nodes(post_xml)
                        post_bounds, _ = _find_like_button_near_expected(
                            post_nodes, cur_scroll_area, target_type, tap_y
                        )
                        if _bounds_close(post_bounds, tap_bounds):
                            print("[TARGET] like button still present near tap (not confirmed)")
                        else:
                            print("[TARGET] like button not found near tap (likely tapped)")
                else:
                    print("[TARGET] no tap bounds resolved; skipping tap")
            else:
                print("[TARGET] missing bounds; skipping tap")

    out = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "llm_provider": os.getenv("LLM_PROVIDER", ""),
            "model": os.getenv("LLM_SMALL_MODEL") or os.getenv("GEMINI_SMALL_MODEL") or "",
            "images_count": llm1_meta.get("images_count"),
            "images_paths": llm1_meta.get("images_paths", []) or photo_paths,
            "timings": {},
            "scoring_ruleset": "long_v0",
        },
        "gate_decision": decision,
        "gate_metrics": {
            "long_score": int(long_score),
            "short_score": int(short_score),
            "long_delta": int(long_score - T_LONG),
            "short_delta": int(short_score - T_SHORT),
            "dom_margin": int(DOM_MARGIN),
            "t_long": int(T_LONG),
            "t_short": int(T_SHORT),
        },
        "manual_override": manual_override,
        "llm3_variant": llm3_variant,
        "llm3_result": llm3_result,
        "llm4_result": llm4_result,
        "target_action": target_action,
        "extracted_profile": extracted,
        "ui_map_summary": {
            "prompts": len(ui_map.get("prompts", [])),
            "photos": len(ui_map.get("photos", [])),
            "poll_options": len(ui_map.get("poll", {}).get("options", [])),
        },
        "profile_eval": eval_result,
        "long_score_result": long_score_result,
        "short_score_result": short_score_result,
        "score_table_long": score_table_long,
        "score_table_short": score_table_short,
        "score_table": score_table,
    }

    if _is_run_json_enabled():
        print(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n" + score_table)

    out_path = ""
    try:
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("logs", f"rating_test_{ts}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        table_path = os.path.join("logs", f"rating_test_{ts}.txt")
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(score_table)
        print(f"Wrote results to {out_path}")
        print(f"Wrote score table to {table_path}")
    except Exception as e:
        print(f"Failed to write results: {e}")

    # SQL logging
    try:
        score_breakdown = (
            f"decision={decision} long_score={long_score} short_score={short_score}\n\n"
            + score_table
        )
        pid = upsert_profile_flat(extracted, eval_result, int(long_score), score_breakdown=score_breakdown)
        if pid is not None:
            update_profile_verdict(pid, decision, "")
            if isinstance(llm3_result, dict) and llm3_result:
                update_profile_opening_messages_json(pid, llm3_result)
            if isinstance(llm4_result, dict) and llm4_result:
                update_profile_opening_pick(pid, llm4_result)
    except Exception as e:
        print(f"[sql] log failed: {e}")

    preference_flag = _classify_preference_flag(long_score, short_score)
    print("\n=== Preference Flag ===")
    print(
        f"classification={preference_flag} "
        f"(long_score={long_score}, short_score={short_score}, "
        "t_long=15, t_short=20, dominance_margin=10)"
    )

    try:
        def _top_contribs(score_result: Dict[str, Any], n: int = 3) -> str:
            contribs = score_result.get("contributions", []) if isinstance(score_result, dict) else []
            items = [
                (abs(int(c.get("delta", 0) or 0)), c)
                for c in contribs
                if int(c.get("delta", 0) or 0) != 0
            ]
            items.sort(key=lambda x: x[0], reverse=True)
            parts = []
            for _, c in items[:n]:
                parts.append(f"{c.get('field','')}: {c.get('value','')} ({c.get('delta','')})")
            return "; ".join(parts) if parts else "none"

        chosen_result = long_score_result if decision == "long_pickup" else short_score_result
        chosen_label = "long" if decision == "long_pickup" else ("short" if decision == "short_pickup" else "n/a")
        summary = (
            f"FINAL decision={decision} "
            f"long_score={long_score} short_score={short_score} "
            f"long_delta={long_score - T_LONG} short_delta={short_score - T_SHORT} "
            f"key_{chosen_label}_contributors={_top_contribs(chosen_result)}"
        )
        print(summary)
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
