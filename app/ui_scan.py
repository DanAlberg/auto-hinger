import os
import time
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image

from helper_functions import swipe, tap
from runtime import _log
from text_utils import normalize_dashes

def _normalize_text_basic(text: str) -> str:
    import re
    s = (text or "").lower()
    s = normalize_dashes(s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())


def _compute_ahash(img: Image.Image, size: int = 8) -> int:
    if img.mode != "L":
        img = img.convert("L")
    resample = getattr(Image, "LANCZOS", 1)
    small = img.resize((size, size), resample)
    pixels = list(small.getdata())
    if not pixels:
        return 0
    avg = sum(pixels) / len(pixels)
    bits = 0
    for i, p in enumerate(pixels):
        if p >= avg:
            bits |= 1 << i
    return bits


def _compute_center_ahash(
    img: Image.Image,
    size: int = 8,
    crop_ratio: float = 0.6,
) -> int:
    """
    Compute aHash on a center crop to reduce UI overlay influence (e.g., like button).
    """
    try:
        w, h = img.size
        side = int(min(w, h) * crop_ratio)
        if side <= 0:
            return _compute_ahash(img, size=size)
        left = max(0, (w - side) // 2)
        top = max(0, (h - side) // 2)
        crop = img.crop((left, top, left + side, top + side))
        return _compute_ahash(crop, size=size)
    except Exception:
        return _compute_ahash(img, size=size)


def _ahash_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _compute_center_ahash_from_file(
    path: str,
    crop_ratio: float = 0.6,
) -> Optional[int]:
    if not path or not os.path.isfile(path):
        return None
    try:
        img = Image.open(path).convert("RGB")
        return _compute_center_ahash(img, crop_ratio=crop_ratio)
    except Exception:
        return None


def _extract_xml_root(raw: str) -> str:
    """
    UIAutomator dumps sometimes include prefix text. Strip to the <hierarchy> root.
    """
    if not raw:
        return ""
    idx = raw.find("<hierarchy")
    if idx == -1:
        return ""
    return raw[idx:]


def _dump_ui_xml(device, tmp_path: str = "/sdcard/hinge_ui.xml") -> str:
    """
    Dump UI hierarchy to a temp path on-device and return the XML string.
    Uses a single rotating file to avoid cluttering the device storage.
    """
    try:
        device.shell(f"uiautomator dump {tmp_path}")
        raw = device.shell(f"cat {tmp_path}")
        # Best-effort cleanup; ignore failures.
        try:
            device.shell(f"rm {tmp_path}")
        except Exception:
            pass
        xml = _extract_xml_root(raw)
        if not xml:
            _log("[UI] Empty/invalid XML dump")
        return xml
    except Exception as e:
        _log(f"[UI] XML dump failed: {e}")
        return ""


def _parse_bounds(bounds: str) -> Optional[Tuple[int, int, int, int]]:
    if not bounds:
        return None
    try:
        left_top, right_bottom = bounds.split("][")
        left_top = left_top.replace("[", "")
        right_bottom = right_bottom.replace("]", "")
        x1, y1 = [int(v) for v in left_top.split(",")]
        x2, y2 = [int(v) for v in right_bottom.split(",")]
        return x1, y1, x2, y2
    except Exception:
        return None


def _bounds_center(bounds: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bounds
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def _bounds_area(bounds: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bounds
    return max(0, x2 - x1) * max(0, y2 - y1)


def _bounds_contains(
    outer: Tuple[int, int, int, int],
    inner: Tuple[int, int, int, int],
) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _find_enclosing_bounds(
    nodes: List[Dict[str, Any]],
    inner: Optional[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    if not inner:
        return None
    inner_area = _bounds_area(inner)
    best_bounds = None
    best_area = None
    for n in nodes:
        b = n.get("bounds")
        if not b or not _bounds_contains(b, inner):
            continue
        area = _bounds_area(b)
        if area <= inner_area:
            continue
        if best_area is None or area < best_area:
            best_area = area
            best_bounds = b
    return best_bounds or inner


def _bounds_close(
    a: Optional[Tuple[int, int, int, int]],
    b: Optional[Tuple[int, int, int, int]],
    tol: int = 24,
) -> bool:
    if not a or not b:
        return False
    ax, ay = _bounds_center(a)
    bx, by = _bounds_center(b)
    return abs(ax - bx) <= tol and abs(ay - by) <= tol


def _parse_prompt_content_desc(content_desc: str) -> Tuple[str, str]:
    cd = (content_desc or "").strip()
    if not cd.startswith("Prompt:"):
        return "", ""
    if "Answer:" not in cd:
        return "", ""
    prompt_part, answer_part = cd.split("Answer:", 1)
    prompt_text = prompt_part.replace("Prompt:", "").strip().strip(".")
    answer_text = answer_part.strip()
    return prompt_text, answer_text


def _find_prompt_bounds_by_text(
    nodes: List[Dict[str, Any]],
    prompt_text: str,
    answer_text: str,
) -> Optional[Tuple[int, int, int, int]]:
    if not prompt_text or not answer_text:
        return None
    target_key = _normalize_text_basic(prompt_text) + "||" + _normalize_text_basic(answer_text)
    for n in nodes:
        cd = (n.get("content_desc") or "").strip()
        if not cd.startswith("Prompt:"):
            continue
        p_txt, a_txt = _parse_prompt_content_desc(cd)
        if not p_txt or not a_txt:
            continue
        key = _normalize_text_basic(p_txt) + "||" + _normalize_text_basic(a_txt)
        if key == target_key:
            return n.get("bounds")
    return None


def _find_poll_option_bounds_by_text(
    nodes: List[Dict[str, Any]],
    option_text: str,
) -> Optional[Tuple[int, int, int, int]]:
    if not option_text:
        return None
    target_norm = _normalize_text_basic(option_text)
    for n in nodes:
        cd = (n.get("content_desc") or "").strip()
        if not cd.startswith("Option:"):
            continue
        opt_text = cd.replace("Option:", "").strip()
        if _normalize_text_basic(opt_text) == target_norm:
            return n.get("bounds")
    return None


def _find_like_button_near_bounds_screen(
    nodes: List[Dict[str, Any]],
    target_bounds: Tuple[int, int, int, int],
    prefer_type: str,
    max_gap: int = 160,
) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
    prefer = (prefer_type or "").lower()
    tb = target_bounds
    candidates: List[Tuple[int, Tuple[int, int, int, int], str]] = []
    fallback: List[Tuple[int, Tuple[int, int, int, int], str]] = []
    for n in nodes:
        if n.get("cls") != "android.widget.Button":
            continue
        cd = (n.get("content_desc") or "").strip()
        if "like" not in cd.lower():
            continue
        b = n.get("bounds")
        if not b:
            continue
        ly = _bounds_center(b)[1]
        if tb[1] <= ly <= tb[3] + max_gap:
            dist = 0 if tb[1] <= ly <= tb[3] else abs(ly - tb[3])
        else:
            dist = abs(ly - tb[3])
        entry = (dist, b, cd)
        if prefer and prefer in cd.lower():
            candidates.append(entry)
        else:
            fallback.append(entry)
    pool = candidates if candidates else fallback
    if not pool:
        return None, ""
    pool.sort(key=lambda x: x[0])
    if pool[0][0] > max_gap:
        return None, ""
    _, best_b, best_cd = pool[0]
    return best_b, best_cd


def _flatten_ui_nodes(root: ET.Element) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []

    def walk(el: ET.Element) -> None:
        attrs = el.attrib or {}
        bounds = _parse_bounds(attrs.get("bounds", ""))
        node = {
            "text": attrs.get("text", "") or "",
            "content_desc": attrs.get("content-desc", "") or "",
            "cls": attrs.get("class", "") or "",
            "scrollable": attrs.get("scrollable", "") == "true",
            "bounds": bounds,
        }
        if bounds:
            nodes.append(node)
        for child in list(el):
            walk(child)

    walk(root)
    return nodes


def _parse_ui_nodes(xml_text: str) -> List[Dict[str, Any]]:
    if not xml_text:
        return []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []
    return _flatten_ui_nodes(root)


def _find_scroll_area(nodes: List[Dict[str, Any]]) -> Optional[Tuple[int, int, int, int]]:
    # Choose the largest scrollable container (by height) as the profile scroll area.
    scroll_nodes = [n for n in nodes if n.get("scrollable") and n.get("bounds")]
    if not scroll_nodes:
        return None
    return max(scroll_nodes, key=lambda n: n["bounds"][3] - n["bounds"][1])["bounds"]


def _find_horizontal_scroll_area(
    nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
) -> Optional[Tuple[int, int, int, int]]:
    """
    Heuristic to find the horizontal biometrics scroller inside the profile.
    """
    top, bottom = scroll_area[1], scroll_area[3]
    candidates: List[Tuple[int, Tuple[int, int, int, int]]] = []
    for n in nodes:
        if not n.get("scrollable"):
            continue
        b = n.get("bounds")
        if not b:
            continue
        # Must be inside the vertical scroll area.
        if b[3] <= top or b[1] >= bottom:
            continue
        w = b[2] - b[0]
        h = b[3] - b[1]
        if h <= 0:
            continue
        # Wide + short = likely horizontal scroller.
        if w >= 600 and (w / h) >= 2.0 and h <= 260:
            candidates.append((w, b))
    if not candidates:
        return None
    # Prefer the widest candidate.
    return sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]


def _find_dislike_bounds(nodes: List[Dict[str, Any]]) -> Optional[Tuple[int, int, int, int]]:
    for n in nodes:
        cd = (n.get("content_desc") or "").strip()
        if not cd:
            continue
        if cd.lower().startswith("skip "):
            return n.get("bounds")
    return None


def _find_add_comment_bounds(nodes: List[Dict[str, Any]]) -> Optional[Tuple[int, int, int, int]]:
    for n in nodes:
        tx = (n.get("text") or "").strip().lower()
        if tx == "add a comment":
            return n.get("bounds")
    for n in nodes:
        b = n.get("bounds")
        if not b:
            continue
        tx = (n.get("text") or "").strip().lower()
        if tx != "add a comment":
            continue
        return b
    return None


def _find_send_priority_like_bounds(nodes: List[Dict[str, Any]]) -> Optional[Tuple[int, int, int, int]]:
    target_norm = _normalize_text_basic("send priority like with message")
    for n in nodes:
        cd = _normalize_text_basic(n.get("content_desc") or "")
        if cd == target_norm:
            return _find_enclosing_bounds(nodes, n.get("bounds"))
        tx = _normalize_text_basic(n.get("text") or "")
        if tx == target_norm:
            return _find_enclosing_bounds(nodes, n.get("bounds"))
    return None


def _find_send_like_anyway_bounds(nodes: List[Dict[str, Any]]) -> Optional[Tuple[int, int, int, int]]:
    target_norm = _normalize_text_basic("send like anyway")
    for n in nodes:
        cd = _normalize_text_basic(n.get("content_desc") or "")
        if cd == target_norm:
            return _find_enclosing_bounds(nodes, n.get("bounds"))
        tx = _normalize_text_basic(n.get("text") or "")
        if tx == target_norm:
            return _find_enclosing_bounds(nodes, n.get("bounds"))
    return None


def _clean_name_text(text: str) -> str:
    return (text or "").strip().strip(" .,!?:;")


def _looks_like_name(text: str) -> bool:
    t = (text or "").strip()
    if not t or len(t) > 40:
        return False
    lower = " ".join(t.lower().split())
    if lower in {"she", "he", "they", "active today", "active now", "active recently", "online"}:
        return False
    if lower.startswith("active "):
        return False
    if any(ch.isdigit() for ch in t):
        return False
    has_alpha = False
    for ch in t:
        if ch.isalpha():
            has_alpha = True
            continue
        if ch in " -'’":
            continue
        return False
    return has_alpha


def _extract_name_from_nodes(
    nodes: List[Dict[str, Any]],
    scroll_area: Optional[Tuple[int, int, int, int]],
) -> str:
    if not nodes:
        return ""
    import re

    for n in nodes:
        cd = (n.get("content_desc") or "").strip()
        if not cd:
            continue
        m = re.match(r"^Skip\s+(.+)$", cd, flags=re.IGNORECASE)
        if m:
            name = _clean_name_text(m.group(1))
            if _looks_like_name(name):
                return name

    for n in nodes:
        cd = (n.get("content_desc") or "").strip()
        if not cd:
            continue
        m = re.match(r"^(.+?)(?:'s|’s)\s+photo$", cd, flags=re.IGNORECASE)
        if m:
            name = _clean_name_text(m.group(1))
            if _looks_like_name(name):
                return name

    if not scroll_area:
        return ""
    top_limit = scroll_area[1]
    candidates: List[Tuple[int, str]] = []
    for n in nodes:
        tx = (n.get("text") or "").strip()
        if not tx:
            continue
        b = n.get("bounds")
        if not b:
            continue
        cy = _bounds_center(b)[1]
        if cy >= top_limit:
            continue
        if not _looks_like_name(tx):
            continue
        candidates.append((cy, _clean_name_text(tx)))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    return ""


def _node_key(node: Dict[str, Any]) -> str:
    # Use content-desc or text to match nodes across scrolls.
    cd = (node.get("content_desc") or "").strip()
    tx = (node.get("text") or "").strip()
    if cd:
        return f"cd:{cd}"
    if tx:
        return f"tx:{tx}"
    return ""


def _compute_scroll_delta(
    prev_nodes: List[Dict[str, Any]],
    curr_nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
) -> Optional[int]:
    """
    Estimate scroll delta by matching stable nodes between two XML dumps.
    Returns positive pixels when scrolling DOWN (content moves up).
    """
    if not prev_nodes or not curr_nodes:
        return None
    top, bottom = scroll_area[1], scroll_area[3]

    def in_scroll(n: Dict[str, Any]) -> bool:
        b = n.get("bounds")
        if not b:
            return False
        return b[1] < bottom and b[3] > top

    prev_map: Dict[str, List[int]] = {}
    for n in prev_nodes:
        if not in_scroll(n):
            continue
        key = _node_key(n)
        if not key or key.startswith("cd:Like"):
            continue
        _, y = _bounds_center(n["bounds"])
        prev_map.setdefault(key, []).append(y)

    deltas: List[int] = []
    for n in curr_nodes:
        if not in_scroll(n):
            continue
        key = _node_key(n)
        if not key or key.startswith("cd:Like"):
            continue
        if key not in prev_map:
            continue
        _, y = _bounds_center(n["bounds"])
        # Match against the closest prior y for this key.
        prev_ys = prev_map.get(key, [])
        if not prev_ys:
            continue
        closest_prev = min(prev_ys, key=lambda py: abs(py - y))
        deltas.append(closest_prev - y)

    if not deltas:
        return None
    deltas.sort()
    return deltas[len(deltas) // 2]


def _screen_signature(
    nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
) -> Set[Tuple[str, int]]:
    """
    Build a coarse signature of visible nodes to detect no-move scrolls.
    """
    top, bottom = scroll_area[1], scroll_area[3]
    sig: Set[Tuple[str, int]] = set()
    for n in nodes:
        b = n.get("bounds")
        if not b:
            continue
        if not (b[1] < bottom and b[3] > top):
            continue
        key = _node_key(n)
        if not key or key.startswith("cd:Like"):
            continue
        _, cy = _bounds_center(b)
        sig.add((key, int(round(cy / 10.0)) * 10))
    return sig


def _annotate_nodes_with_abs_bounds(
    nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
    scroll_offset: int,
) -> List[Dict[str, Any]]:
    """
    Convert on-screen bounds to absolute content bounds by adding scroll_offset
    for nodes that fall inside the scrollable profile area.
    """
    top, bottom = scroll_area[1], scroll_area[3]
    annotated: List[Dict[str, Any]] = []
    for n in nodes:
        b = n.get("bounds")
        if not b:
            continue
        in_scroll = b[1] < bottom and b[3] > top
        abs_bounds = (b[0], b[1] + scroll_offset, b[2], b[3] + scroll_offset) if in_scroll else b
        nn = dict(n)
        nn["in_scroll"] = in_scroll
        nn["abs_bounds"] = abs_bounds
        annotated.append(nn)
    return annotated


_BIOMETRIC_LABEL_MAP = {
    "age": "Age",
    "gender": "Gender",
    "sexuality": "Sexuality",
    "height": "Height",
    "job": "Job title",
    "job title": "Job title",
    "college or university": "University",
    "university": "University",
    "religion": "Religious Beliefs",
    "home town": "Home town",
    "hometown": "Home town",
    "languages spoken": "Languages spoken",
    "ethnicity": "Explicit Ethnicity",
    "dating intentions": "Dating Intentions",
    "relationship type": "Relationship type",
    "children": "Children",
    "family plans": "Family plans",
    "covid vaccine": "Covid Vaccine",
    "politics": "Politics",
    "zodiac sign": "Zodiac Sign",
    "pets": "Pets",
    "drinking": "Drinking",
    "smoking": "Smoking",
    "marijuana": "Marijuana",
    "drugs": "Drugs",
    "location": "Location",
}
def _normalize_label(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _parse_height_value(raw: str) -> Optional[int]:
    if not raw:
        return None
    s = raw.strip().lower()
    # If explicitly in cm (or looks like cm), take first number.
    import re
    nums = [int(n) for n in re.findall(r"\d+", s)]
    if not nums:
        return None
    if "cm" in s or (nums and nums[0] >= 100):
        return nums[0]
    # Handle feet/inches formats like 5'7 or 5 ft 7 in.
    if "'" in s or "ft" in s:
        feet = nums[0] if nums else 0
        inches = nums[1] if len(nums) > 1 else 0
        cm = int(round(feet * 30.48 + inches * 2.54))
        return cm
    # Fallback: if two small numbers, treat as feet/inches.
    if len(nums) >= 2 and nums[0] <= 7 and nums[1] <= 11:
        cm = int(round(nums[0] * 30.48 + nums[1] * 2.54))
        return cm
    return nums[0]


def _extract_biometrics_from_nodes(
    nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
) -> Dict[str, Any]:
    """
    Extract biometrics visible on the current screen by pairing label nodes
    (content-desc) with value text nodes to their right.
    """
    top, bottom = scroll_area[1], scroll_area[3]

    def in_scroll(n: Dict[str, Any]) -> bool:
        b = n.get("bounds")
        if not b:
            return False
        return b[1] < bottom and b[3] > top

    label_nodes: List[Dict[str, Any]] = []
    value_nodes: List[Dict[str, Any]] = []
    for n in nodes:
        if not in_scroll(n):
            continue
        cd = n.get("content_desc") or ""
        tx = n.get("text") or ""
        if cd:
            label_nodes.append(n)
        if tx:
            value_nodes.append(n)

    updates: Dict[str, Any] = {}
    for label in label_nodes:
        label_text = _normalize_label(label.get("content_desc", ""))
        if label_text not in _BIOMETRIC_LABEL_MAP:
            continue
        field = _BIOMETRIC_LABEL_MAP[label_text]
        lb = label["bounds"]
        _, ly = _bounds_center(lb)
        best_val = None
        best_score = None
        for val in value_nodes:
            vb = val["bounds"]
            # Value should be to the right of the label and roughly aligned vertically.
            if vb[0] < lb[2] - 5:
                continue
            _, vy = _bounds_center(vb)
            if abs(vy - ly) > max(60, (lb[3] - lb[1]) * 1.5):
                continue
            dx = vb[0] - lb[2]
            score = abs(vy - ly) * 2 + dx
            if best_score is None or score < best_score:
                best_score = score
                best_val = val
        if not best_val:
            continue
        raw_val = (best_val.get("text") or "").strip()
        if not raw_val:
            continue
        if field == "Age":
            try:
                updates[field] = int("".join(ch for ch in raw_val if ch.isdigit()))
            except Exception:
                continue
        elif field == "Height":
            height_val = _parse_height_value(raw_val)
            if height_val is not None:
                updates[field] = height_val
        else:
            updates[field] = raw_val
    return updates


def _hscroll_once(
    device,
    area: Tuple[int, int, int, int],
    direction: str = "left",
    distance_px: Optional[int] = None,
    duration_ms: int = 350,
) -> None:
    left, top, right, bottom = area
    mid_y = int((top + bottom) / 2)
    area_w = right - left
    dist = int(distance_px or (area_w * 0.6))
    dist = max(80, min(dist, int(area_w * 0.9)))
    if direction == "left":
        x_start = int(right - area_w * 0.1)
        x_end = max(int(left + area_w * 0.1), x_start - dist)
    else:
        x_start = int(left + area_w * 0.1)
        x_end = min(int(right - area_w * 0.1), x_start + dist)
    swipe(device, x_start, mid_y, x_end, mid_y, duration_ms)
    _log(f"[BIOMETRICS] hscroll {direction} x={x_start}->{x_end} y={mid_y}")


def _scan_biometrics_hscroll(
    device,
    nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
    biometrics: Dict[str, Any],
    max_swipes: int = 12,
) -> List[Dict[str, Any]]:
    """
    Attempt to reveal additional biometrics by horizontal scrolling.
    """
    h_area = _find_horizontal_scroll_area(nodes, scroll_area)
    if not h_area:
        _log("[BIOMETRICS] no horizontal scroller detected")
        return nodes
    _log(f"[BIOMETRICS] horizontal scroller area={h_area}")
    swipes_done = 0
    no_new = 0
    while swipes_done < max_swipes and no_new < 2:
        _hscroll_once(device, h_area, "left")
        time.sleep(0.4)
        xml = _dump_ui_xml(device)
        nodes = _parse_ui_nodes(xml)
        updates = _extract_biometrics_from_nodes(nodes, scroll_area)
        new_any = False
        for k, v in updates.items():
            if k not in biometrics and v not in ("", None):
                biometrics[k] = v
                _log(f"[BIOMETRICS] {k} = {v}")
                new_any = True
        if new_any:
            no_new = 0
        else:
            no_new += 1
        swipes_done += 1
    return nodes


def _add_or_update_by_abs_y(
    items: List[Dict[str, Any]],
    new_item: Dict[str, Any],
    y_key: str = "abs_center_y",
    tol: int = 20,
    dedupe_key: Optional[str] = None,
) -> None:
    if dedupe_key:
        for item in items:
            if item.get(dedupe_key) == new_item.get(dedupe_key):
                # Update bounds if missing or more complete.
                if not item.get("abs_bounds") and new_item.get("abs_bounds"):
                    item["abs_bounds"] = new_item["abs_bounds"]
                return
    new_y = new_item.get(y_key)
    if new_y is None:
        items.append(new_item)
        return
    for item in items:
        if abs((item.get(y_key) or 0) - new_y) <= tol:
            # Prefer larger bounds (more complete photo).
            if new_item.get("abs_bounds") and item.get("abs_bounds"):
                ob = item["abs_bounds"]
                nb = new_item["abs_bounds"]
                if (nb[3] - nb[1]) > (ob[3] - ob[1]):
                    item["abs_bounds"] = nb
            return
    items.append(new_item)


def _nearest_like_bounds(
    card_bounds: Tuple[int, int, int, int],
    likes: List[Dict[str, Any]],
    max_gap: int = 180,
) -> Optional[Tuple[int, int, int, int]]:
    if not likes:
        return None
    cb = card_bounds
    candidates: List[Tuple[int, Tuple[int, int, int, int]]] = []
    for l in likes:
        lb = l.get("abs_bounds")
        if not lb:
            continue
        ly = _bounds_center(lb)[1]
        # Prefer likes vertically inside the card bounds (or just below).
        if cb[1] <= ly <= cb[3] + max_gap:
            dist = 0 if cb[1] <= ly <= cb[3] else abs(ly - cb[3])
        else:
            dist = abs(ly - cb[3])
        candidates.append((dist, lb))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    # Only accept if reasonably close.
    if candidates[0][0] > max_gap:
        return None
    return candidates[0][1]


def _update_ui_map_text_only(
    ui_map: Dict[str, Any],
    nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
    scroll_offset: int,
) -> None:
    """
    Update UI map with prompts, polls, and like buttons only (no photos).
    """
    annotated = _annotate_nodes_with_abs_bounds(nodes, scroll_area, scroll_offset)
    top, bottom = scroll_area[1], scroll_area[3]

    # Collect like buttons visible on this screen first.
    screen_likes: List[Dict[str, Any]] = []
    for n in annotated:
        if not n.get("in_scroll"):
            continue
        cd = (n.get("content_desc") or "").strip()
        cls = n.get("cls") or ""
        b = n.get("abs_bounds")
        if not b:
            continue
        _, cy = _bounds_center(b)
        if cls == "android.widget.Button" and cd.lower().startswith("like"):
            cd_lower = cd.lower()
            if "photo" in cd_lower:
                like_type = "photo"
            elif "prompt" in cd_lower:
                like_type = "prompt"
            else:
                like_type = "unknown"
            screen_likes.append(
                {
                    "type": like_type,
                    "abs_bounds": b,
                    "abs_center_y": cy,
                    "content_desc": cd,
                }
            )

    screen_prompt_likes = [l for l in screen_likes if l.get("type") == "prompt"]

    for l in screen_likes:
        _add_or_update_by_abs_y(
            ui_map["likes"],
            {
                "type": l["type"],
                "abs_bounds": l["abs_bounds"],
                "abs_center_y": l["abs_center_y"],
                "content_desc": l.get("content_desc", ""),
            },
            tol=10,
        )

    for n in annotated:
        if not n.get("in_scroll"):
            continue
        cd = (n.get("content_desc") or "").strip()
        b = n.get("abs_bounds")
        if not b:
            continue
        _, cy = _bounds_center(b)

        if cd.startswith("Prompt:"):
            if "Answer:" in cd:
                prompt_part, answer_part = cd.split("Answer:", 1)
                prompt_text = prompt_part.replace("Prompt:", "").strip().strip(".")
                answer_text = answer_part.strip()
                key = f"{prompt_text}||{answer_text}"
                prompt_like = _nearest_like_bounds(b, screen_prompt_likes, max_gap=120)
                prompt_item = {
                    "prompt": prompt_text,
                    "answer": answer_text,
                    "abs_bounds": b,
                    "abs_center_y": cy,
                    "key": key,
                }
                if prompt_like:
                    prompt_item["like_bounds"] = prompt_like
                _add_or_update_by_abs_y(ui_map["prompts"], prompt_item, dedupe_key="key")
            else:
                question = cd.replace("Prompt:", "").strip()
                if question and not ui_map["poll"].get("question"):
                    ui_map["poll"]["question"] = question

        if cd.startswith("Option:"):
            option_text = cd.replace("Option:", "").strip()
            if option_text:
                _add_or_update_by_abs_y(
                    ui_map["poll"]["options"],
                    {
                        "text": option_text,
                        "abs_bounds": b,
                        "abs_center_y": cy,
                        "key": option_text,
                    },
                    dedupe_key="key",
                )


def _find_primary_photo_bounds(
    nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
) -> Optional[Tuple[int, int, int, int]]:
    top, bottom = scroll_area[1], scroll_area[3]
    best = None
    best_area = None
    for n in nodes:
        if n.get("cls") != "android.widget.ImageView":
            continue
        cd = (n.get("content_desc") or "").lower()
        if "photo" not in cd:
            continue
        b = n.get("bounds")
        if not b:
            continue
        if b[1] >= bottom or b[3] <= top:
            continue
        w = b[2] - b[0]
        h = b[3] - b[1]
        if w < 200 or h < 200:
            continue
        area = w * h
        if best_area is None or area > best_area:
            best_area = area
            best = b
    return best


def _find_visible_photo_bounds_all(
    nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
) -> List[Tuple[int, int, int, int]]:
    top, bottom = scroll_area[1], scroll_area[3]
    results: List[Tuple[int, int, int, int]] = []
    for n in nodes:
        if n.get("cls") != "android.widget.ImageView":
            continue
        cd = (n.get("content_desc") or "").lower()
        if "photo" not in cd:
            continue
        b = n.get("bounds")
        if not b:
            continue
        if b[1] >= bottom or b[3] <= top:
            continue
        results.append(b)
    return results


def _clamp_bounds_to_screen(
    bounds: Tuple[int, int, int, int],
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = bounds
    x1 = max(0, min(width - 1, x1))
    x2 = max(1, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(1, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

def _is_square_bounds(bounds: Tuple[int, int, int, int], tol: int = 12) -> bool:
    w = bounds[2] - bounds[0]
    h = bounds[3] - bounds[1]
    return abs(w - h) <= tol


def _compute_center_ahash_from_bounds(
    device,
    bounds: Tuple[int, int, int, int],
    width: int,
    height: int,
    crop_ratio: float = 0.6,
) -> Optional[int]:
    cb = _clamp_bounds_to_screen(bounds, width, height)
    if not cb:
        return None
    try:
        img_bytes = device.screencap()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        crop = img.crop(cb)
        return _compute_center_ahash(crop, crop_ratio=crop_ratio)
    except Exception:
        return None


def _match_photo_bounds_by_hash(
    device,
    width: int,
    height: int,
    nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
    target_hash: int,
    expected_screen_y: Optional[int] = None,
    max_dist: int = 18,
    square_only: bool = True,
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[int]]:
    candidates = _find_visible_photo_bounds_all(nodes, scroll_area)
    if not candidates:
        return None, None
    if square_only:
        square_candidates = [b for b in candidates if _is_square_bounds(b)]
        if square_candidates:
            candidates = square_candidates
        else:
            _log("[TARGET] no square photo candidates; retrying with partials")
    if expected_screen_y is not None:
        candidates.sort(key=lambda b: abs(_bounds_center(b)[1] - expected_screen_y))
    img_bytes = device.screencap()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    best_bounds = None
    best_dist = None
    for b in candidates:
        cb = _clamp_bounds_to_screen(b, width, height)
        if not cb:
            continue
        crop = img.crop(cb)
        h = _compute_center_ahash(crop)
        dist = _ahash_distance(h, target_hash)
        _log(f"[TARGET] photo hash candidate bounds={cb} dist={dist}")
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_bounds = cb
    if best_dist is None:
        return None, None
    if best_dist <= max_dist:
        return best_bounds, best_dist
    return None, best_dist


def _find_like_button_in_photo(
    nodes: List[Dict[str, Any]],
    photo_bounds: Tuple[int, int, int, int],
) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
    x1, y1, x2, y2 = photo_bounds
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    best = None
    best_score = None
    best_desc = ""
    for n in nodes:
        if n.get("cls") != "android.widget.Button":
            continue
        cd = (n.get("content_desc") or "").strip()
        if "like" not in cd.lower():
            continue
        b = n.get("bounds")
        if not b:
            continue
        cx, cy = _bounds_center(b)
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            continue
        score = (cx - mid_x) + (cy - mid_y)
        if best_score is None or score > best_score:
            best_score = score
            best = b
            best_desc = cd
    if best:
        return best, best_desc

    # Fallback: nearest like button to bottom-right corner.
    br_x, br_y = x2, y2
    fallback = None
    fallback_desc = ""
    fallback_dist = None
    for n in nodes:
        if n.get("cls") != "android.widget.Button":
            continue
        cd = (n.get("content_desc") or "").strip()
        if "like" not in cd.lower():
            continue
        b = n.get("bounds")
        if not b:
            continue
        cx, cy = _bounds_center(b)
        dist = abs(cx - br_x) + abs(cy - br_y)
        if fallback_dist is None or dist < fallback_dist:
            fallback_dist = dist
            fallback = b
            fallback_desc = cd
    return fallback, fallback_desc


def _find_like_button_near_expected(
    nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
    target_type: str,
    expected_screen_y: int,
) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
    """
    Find the closest like button on screen to the expected Y.
    Prefer matching content-desc for prompt/photo when possible.
    """
    top, bottom = scroll_area[1], scroll_area[3]
    target_type = (target_type or "").strip().lower()
    prefer = ""
    if target_type == "photo":
        prefer = "photo"
    elif target_type == "prompt":
        prefer = "prompt"

    candidates: List[Tuple[int, Tuple[int, int, int, int], str]] = []
    fallback: List[Tuple[int, Tuple[int, int, int, int], str]] = []
    for n in nodes:
        if n.get("cls") != "android.widget.Button":
            continue
        cd = (n.get("content_desc") or "").strip()
        if "like" not in cd.lower():
            continue
        b = n.get("bounds")
        if not b:
            continue
        if b[1] >= bottom or b[3] <= top:
            continue
        cy = _bounds_center(b)[1]
        dist = abs(cy - expected_screen_y)
        entry = (dist, b, cd)
        if prefer and prefer in cd.lower():
            candidates.append(entry)
        else:
            fallback.append(entry)

    pool = candidates if candidates else fallback
    if not pool:
        return None, ""
    pool.sort(key=lambda x: x[0])
    _, best_b, best_cd = pool[0]
    return best_b, best_cd


def _ensure_photo_square(
    device,
    width: int,
    height: int,
    scroll_area: Tuple[int, int, int, int],
    nodes: List[Dict[str, Any]],
    offset: int,
    photo_bounds: Tuple[int, int, int, int],
    max_attempts: int = 4,
    target_abs_center_y: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], int, Optional[Tuple[int, int, int, int]]]:
    vb = photo_bounds
    vb_w = vb[2] - vb[0]
    vb_h = vb[3] - vb[1]
    is_square = _is_square_bounds(vb)
    attempts = 0
    while not is_square and attempts < max_attempts:
        top_clipped = vb[1] <= scroll_area[1] + 3
        bottom_clipped = vb[3] >= scroll_area[3] - 3
        direction = None
        if top_clipped:
            # At absolute top, we can't scroll up further; reveal bottom instead.
            direction = "down" if offset <= 0 else "up"
        elif bottom_clipped:
            direction = "down"
        if not direction:
            break
        prev_nodes = nodes
        nodes, delta = _scroll_and_capture(
            device,
            width,
            height,
            scroll_area,
            direction,
            prev_nodes,
            distance_px=140,
        )
        offset += delta
        if target_abs_center_y is not None:
            expected_screen_y = int(target_abs_center_y - offset)
            vb = _find_visible_photo_bounds(nodes, scroll_area, expected_screen_y)
        else:
            vb = _find_primary_photo_bounds(nodes, scroll_area)
        if not vb:
            break
        vb_w = vb[2] - vb[0]
        vb_h = vb[3] - vb[1]
        is_square = _is_square_bounds(vb)
        _log(
            f"[PHOTO] micro-scroll {attempts+1} dir={direction} size={vb_w}x{vb_h} square={'yes' if is_square else 'no'}"
        )
        attempts += 1
    return nodes, offset, vb

def _assign_like_buttons(ui_map: Dict[str, Any]) -> None:
    """
    Attach the nearest like button to each prompt/photo card.
    """
    likes = ui_map.get("likes", [])
    for l in likes:
        l["used"] = False

    def assign(cards: List[Dict[str, Any]], prefer_type: str) -> None:
        for card in cards:
            if card.get("like_bounds"):
                continue
            cb = card.get("abs_bounds")
            if not cb:
                continue
            cx, cy = _bounds_center(cb)
            candidates = [
                l for l in likes
                if not l.get("used") and l.get("abs_bounds")
                and l.get("type") == prefer_type
            ]
            if not candidates:
                continue
            def dist(like: Dict[str, Any]) -> int:
                ly = _bounds_center(like["abs_bounds"])[1]
                if cb[1] <= ly <= cb[3]:
                    return 0
                return min(abs(ly - cb[1]), abs(ly - cb[3]))
            best = min(candidates, key=dist)
            best["used"] = True
            card["like_bounds"] = best["abs_bounds"]
            card["like_desc"] = best.get("content_desc", "")

    assign(ui_map.get("prompts", []), "prompt")
    assign(ui_map.get("photos", []), "photo")


def _assign_ids(ui_map: Dict[str, Any]) -> None:
    # Sort top-to-bottom for stable IDs.
    ui_map["prompts"].sort(key=lambda p: p.get("abs_center_y", 0))
    for idx, prompt in enumerate(ui_map["prompts"], start=1):
        prompt["id"] = f"prompt_{idx}"

    ui_map["photos"].sort(key=lambda p: p.get("abs_center_y", 0))
    for idx, photo in enumerate(ui_map["photos"], start=1):
        photo["id"] = f"photo_{idx}"

    ui_map["poll"]["options"].sort(key=lambda o: o.get("abs_center_y", 0))
    for idx, opt in enumerate(ui_map["poll"]["options"], start=1):
        suffix = chr(ord("a") + idx - 1)
        opt["id"] = f"poll_1_{suffix}"


def _scroll_once(
    device,
    width: int,
    height: int,
    scroll_area: Tuple[int, int, int, int],
    direction: str = "down",
    distance_px: Optional[int] = None,
    duration_ms: int = 450,
) -> int:
    """
    Perform a single scroll gesture.
    direction="down" means moving down the profile (content moves up).
    Returns the expected scroll delta in pixels (positive for down, negative for up).
    """
    left, top, right, bottom = scroll_area
    area_h = bottom - top
    x = int((left + right) / 2)
    dist = int(distance_px or (area_h * 0.6))
    dist = max(80, min(dist, int(area_h * 0.9)))

    if direction == "down":
        # Finger swipes up.
        y_start = int(bottom - area_h * 0.15)
        y_end = max(int(top + area_h * 0.1), y_start - dist)
        swipe(device, x, y_start, x, y_end, duration_ms)
        expected = y_start - y_end
    else:
        # direction == "up": finger swipes down.
        y_start = int(top + area_h * 0.15)
        y_end = min(int(bottom - area_h * 0.1), y_start + dist)
        swipe(device, x, y_start, x, y_end, duration_ms)
        expected = -(y_end - y_start)

    _log(f"[SCROLL] {direction} swipe y={y_start}->{y_end} expected_delta={expected}")
    return expected


def _scroll_and_capture(
    device,
    width: int,
    height: int,
    scroll_area: Tuple[int, int, int, int],
    direction: str,
    prev_nodes: List[Dict[str, Any]],
    distance_px: Optional[int] = None,
    duration_ms: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    expected = _scroll_once(
        device,
        width,
        height,
        scroll_area,
        direction,
        distance_px,
        duration_ms=duration_ms or 450,
    )
    time.sleep(0.5)
    xml = _dump_ui_xml(device)
    nodes = _parse_ui_nodes(xml)
    current_scroll_area = _find_scroll_area(nodes) or scroll_area
    actual = _compute_scroll_delta(prev_nodes, nodes, current_scroll_area)
    prev_sig = _screen_signature(prev_nodes, current_scroll_area)
    curr_sig = _screen_signature(nodes, current_scroll_area)
    screen_changed = prev_sig != curr_sig
    overlap = 0.0
    if prev_sig:
        overlap = len(prev_sig & curr_sig) / max(1, len(prev_sig))
    if actual is None:
        if screen_changed:
            actual = expected
            _log(f"[SCROLL] delta=unknown fallback={actual}")
        else:
            actual = 0
            _log("[SCROLL] no-move confirmed (signature unchanged)")
    else:
        # Guard against sign flips.
        if (direction == "down" and actual < 0) or (direction == "up" and actual > 0):
            if screen_changed:
                _log(f"[SCROLL] delta sign mismatch ({actual}); using fallback {expected}")
                actual = expected
            else:
                _log("[SCROLL] sign mismatch but signature unchanged; treating as no-move")
                actual = 0
        elif abs(actual) <= 5 and abs(expected) >= 200:
            if overlap >= 0.80:
                _log(f"[SCROLL] no-move confirmed (overlap={overlap:.2f})")
                actual = 0
            else:
                if screen_changed:
                    _log(f"[SCROLL] delta unreliable (0, overlap={overlap:.2f}); using fallback {expected}")
                    actual = expected
                else:
                    _log("[SCROLL] delta unreliable but signature unchanged; treating as no-move")
                    actual = 0
        elif abs(actual - expected) >= 400 and overlap < 0.60:
            if screen_changed:
                _log(f"[SCROLL] delta mismatch actual={actual} expected={expected} overlap={overlap:.2f}; using fallback")
                actual = expected
            else:
                _log("[SCROLL] delta mismatch but signature unchanged; treating as no-move")
                actual = 0
        else:
            _log(f"[SCROLL] delta=measured {actual}")
    return nodes, actual


def _scroll_to_top(
    device,
    width: int,
    height: int,
    max_attempts: int = 8,
) -> Tuple[List[Dict[str, Any]], Optional[Tuple[int, int, int, int]]]:
    """
    Best-effort reset to the top by scrolling up until no movement.
    Returns the last nodes and scroll_area.
    """
    xml = _dump_ui_xml(device)
    nodes = _parse_ui_nodes(xml)
    scroll_area = _find_scroll_area(nodes)
    if not scroll_area:
        return nodes, None
    no_move = 0
    attempts = 0
    while attempts < max_attempts and no_move < 2:
        prev_nodes = nodes
        nodes, delta = _scroll_and_capture(
            device, width, height, scroll_area, "up", prev_nodes
        )
        if abs(delta) <= 5:
            no_move += 1
        else:
            no_move = 0
        attempts += 1
    _log("[SCROLL] top reset complete")
    return nodes, scroll_area


def _seek_target_on_screen(
    device,
    width: int,
    height: int,
    scroll_area: Tuple[int, int, int, int],
    current_offset: int,
    target_type: str,
    target_info: Dict[str, Any],
    desired_offset: int,
    max_steps: int = 12,
) -> Dict[str, Any]:
    """
    Scroll in small steps until the target is visible. This avoids overshooting
    when offset deltas are unreliable.
    Returns a dict with updated nodes/scroll_area/offset and any found bounds.
    """
    offset = current_offset
    xml = _dump_ui_xml(device)
    nodes = _parse_ui_nodes(xml)
    scroll_area = _find_scroll_area(nodes) or scroll_area
    area_h = max(1, scroll_area[3] - scroll_area[1])
    no_move = 0
    steps = 0

    while steps < max_steps:
        found: Dict[str, Any] = {}
        if target_type == "prompt":
            prompt_bounds = _find_prompt_bounds_by_text(
                nodes,
                target_info.get("prompt", ""),
                target_info.get("answer", ""),
            )
            if prompt_bounds:
                _log(f"[SEEK] prompt visible at {prompt_bounds}")
                found["prompt_bounds"] = prompt_bounds
        elif target_type == "poll":
            option_bounds = _find_poll_option_bounds_by_text(
                nodes, target_info.get("option_text", "")
            )
            if option_bounds:
                _log(f"[SEEK] poll option visible at {option_bounds}")
                found["poll_bounds"] = option_bounds
        elif target_type == "photo":
            target_hash = target_info.get("photo_hash")
            target_photo_bounds = target_info.get("photo_bounds")
            target_abs_center_y = None
            expected_screen_y = None
            if target_photo_bounds:
                target_abs_center_y = int(
                    (target_photo_bounds[1] + target_photo_bounds[3]) / 2
                )
                expected_screen_y = int(target_abs_center_y - offset)
            if target_hash is not None:
                match_bounds, dist = _match_photo_bounds_by_hash(
                    device,
                    width,
                    height,
                    nodes,
                    scroll_area,
                    int(target_hash),
                    expected_screen_y=expected_screen_y,
                    max_dist=18,
                    square_only=True,
                )
                if match_bounds:
                    _log(f"[SEEK] photo hash visible at {match_bounds} dist={dist}")
                    found["photo_match_bounds"] = match_bounds
            if not found and expected_screen_y is not None:
                photo_bounds = _find_visible_photo_bounds(
                    nodes, scroll_area, expected_screen_y
                )
                if photo_bounds and _is_square_bounds(photo_bounds):
                    _log(f"[SEEK] photo candidate near expected y at {photo_bounds}")
                    found["photo_bounds"] = photo_bounds

        if found:
            return {
                "nodes": nodes,
                "scroll_area": scroll_area,
                "scroll_offset": offset,
                **found,
            }

        # Not visible yet; move toward desired offset in smaller steps.
        delta = desired_offset - offset
        if abs(delta) <= 20:
            step_px = 140
        else:
            step_px = min(abs(delta), int(area_h * 0.45))
        direction = "down" if delta > 0 else "up"
        prev_nodes = nodes
        nodes, actual = _scroll_and_capture(
            device,
            width,
            height,
            scroll_area,
            direction,
            prev_nodes,
            distance_px=step_px,
            duration_ms=420,
        )
        scroll_area = _find_scroll_area(nodes) or scroll_area
        offset += actual
        _log(f"[SEEK] step offset now {offset} (target {desired_offset})")
        steps += 1
        if abs(actual) <= 5:
            no_move += 1
        else:
            no_move = 0
        if no_move >= 2:
            _log("[SEEK] no-move twice; stopping")
            break

    return {
        "nodes": nodes,
        "scroll_area": scroll_area,
        "scroll_offset": offset,
    }


def _seek_photo_by_index(
    device,
    width: int,
    height: int,
    scroll_area: Tuple[int, int, int, int],
    target_index: int,
    target_hash: Optional[int] = None,
    max_steps: int = 25,
    max_dist: int = 18,
) -> Dict[str, Any]:
    """
    Photo-only re-acquire by index: scroll from top and count square photos.
    Uses center-hash to avoid double-counting the same photo.
    """
    nodes, scroll_area = _scroll_to_top(device, width, height)
    if not scroll_area:
        return {"nodes": nodes, "scroll_area": scroll_area}
    offset = 0
    area_h = max(1, scroll_area[3] - scroll_area[1])
    step_px = int(area_h * 0.45)
    no_move = 0
    steps = 0
    count = 0
    last_hash: Optional[int] = None

    while steps < max_steps:
        photo_bounds = _find_primary_photo_bounds(nodes, scroll_area)
        if photo_bounds:
            nodes, offset, photo_bounds = _ensure_photo_square(
                device,
                width,
                height,
                scroll_area,
                nodes,
                offset,
                photo_bounds,
            )
            scroll_area = _find_scroll_area(nodes) or scroll_area
            if photo_bounds and _is_square_bounds(photo_bounds):
                h = _compute_center_ahash_from_bounds(
                    device, photo_bounds, width, height
                )
                dist = _ahash_distance(h, target_hash) if (h is not None and target_hash is not None) else None
                _log(f"[SEEK-PHOTO] candidate bounds={photo_bounds} dist={dist}")
                is_new = True
                if h is not None and last_hash is not None:
                    if _ahash_distance(h, last_hash) <= 6:
                        is_new = False
                if is_new:
                    count += 1
                    last_hash = h
                    _log(f"[SEEK-PHOTO] count={count} target={target_index}")
                    if count == target_index:
                        like_bounds, like_desc = _find_like_button_in_photo(
                            nodes, photo_bounds
                        )
                        if like_bounds:
                            if dist is not None and dist > max_dist:
                                _log(f"[SEEK-PHOTO] warning: hash dist {dist} > {max_dist} at target index")
                            _log(
                                f"[SEEK-PHOTO] index match like_bounds={like_bounds}"
                            )
                            return {
                                "nodes": nodes,
                                "scroll_area": scroll_area,
                                "scroll_offset": offset,
                                "tap_bounds": like_bounds,
                                "tap_desc": like_desc,
                            }

        prev_nodes = nodes
        nodes, delta = _scroll_and_capture(
            device,
            width,
            height,
            scroll_area,
            "down",
            prev_nodes,
            distance_px=step_px,
            duration_ms=420,
        )
        scroll_area = _find_scroll_area(nodes) or scroll_area
        offset += delta
        _log(f"[SEEK-PHOTO] step {steps+1} offset={offset}")
        steps += 1
        if abs(delta) <= 5:
            no_move += 1
        else:
            no_move = 0
        if no_move >= 2:
            _log("[SEEK-PHOTO] no-move twice; stopping")
            break

    return {"nodes": nodes, "scroll_area": scroll_area, "scroll_offset": offset}


def _seek_photo_by_index_from_bottom(
    device,
    width: int,
    height: int,
    scroll_area: Tuple[int, int, int, int],
    nodes: Optional[List[Dict[str, Any]]],
    current_offset: int,
    target_index: int,
    total_photos: int,
    target_hash: Optional[int] = None,
    max_steps: int = 25,
    max_dist: int = 18,
) -> Dict[str, Any]:
    """
    Photo re-acquire by index starting from bottom (current screen).
    Counts square photos while scrolling up to reach target from bottom.
    """
    if not scroll_area or total_photos <= 0 or target_index <= 0:
        return {"nodes": nodes, "scroll_area": scroll_area, "scroll_offset": current_offset}
    target_from_bottom = total_photos - target_index + 1
    if target_from_bottom <= 0:
        return {"nodes": nodes, "scroll_area": scroll_area, "scroll_offset": current_offset}
    _log(f"[SEEK-PHOTO] reverse target_from_bottom={target_from_bottom} total={total_photos}")

    offset = current_offset
    if not nodes:
        xml = _dump_ui_xml(device)
        nodes = _parse_ui_nodes(xml)
    scroll_area = _find_scroll_area(nodes) or scroll_area
    area_h = max(1, scroll_area[3] - scroll_area[1])
    step_px = int(area_h * 0.45)
    no_move = 0
    steps = 0
    count = 0
    seen_hashes: List[int] = []

    while steps < max_steps:
        photo_bounds = _find_primary_photo_bounds(nodes, scroll_area)
        if photo_bounds:
            nodes, offset, photo_bounds = _ensure_photo_square(
                device,
                width,
                height,
                scroll_area,
                nodes,
                offset,
                photo_bounds,
            )
            scroll_area = _find_scroll_area(nodes) or scroll_area
            if photo_bounds and _is_square_bounds(photo_bounds):
                h = _compute_center_ahash_from_bounds(
                    device, photo_bounds, width, height
                )
                dist = _ahash_distance(h, target_hash) if (h is not None and target_hash is not None) else None
                _log(f"[SEEK-PHOTO] candidate bounds={photo_bounds} dist={dist}")
                is_new = True
                if h is not None:
                    for prev in seen_hashes:
                        if _ahash_distance(h, prev) <= 6:
                            is_new = False
                            break
                if is_new:
                    count += 1
                    if h is not None:
                        seen_hashes.append(h)
                    _log(f"[SEEK-PHOTO] count={count} target={target_from_bottom}")
                    if count == target_from_bottom:
                        like_bounds, like_desc = _find_like_button_in_photo(
                            nodes, photo_bounds
                        )
                        if like_bounds:
                            if dist is not None and dist > max_dist:
                                _log(
                                    f"[SEEK-PHOTO] warning: hash dist {dist} > {max_dist} at target index"
                                )
                            _log(
                                f"[SEEK-PHOTO] index match like_bounds={like_bounds}"
                            )
                            return {
                                "nodes": nodes,
                                "scroll_area": scroll_area,
                                "scroll_offset": offset,
                                "tap_bounds": like_bounds,
                                "tap_desc": like_desc,
                            }

        prev_nodes = nodes
        nodes, delta = _scroll_and_capture(
            device,
            width,
            height,
            scroll_area,
            "up",
            prev_nodes,
            distance_px=step_px,
            duration_ms=420,
        )
        scroll_area = _find_scroll_area(nodes) or scroll_area
        offset += delta
        _log(f"[SEEK-PHOTO] step {steps+1} offset={offset}")
        steps += 1
        if abs(delta) <= 5:
            no_move += 1
        else:
            no_move = 0
        if no_move >= 2:
            _log("[SEEK-PHOTO] no-move twice; stopping")
            break

    return {"nodes": nodes, "scroll_area": scroll_area, "scroll_offset": offset}


def _compute_desired_offset(
    abs_bounds: Tuple[int, int, int, int],
    scroll_area: Tuple[int, int, int, int],
) -> int:
    """
    Compute a scroll_offset that keeps the target bounds fully visible.
    """
    top, bottom = scroll_area[1], scroll_area[3]
    abs_top, abs_bottom = abs_bounds[1], abs_bounds[3]
    scroll_center = int((top + bottom) / 2)
    target_center = int((abs_top + abs_bottom) / 2)
    desired = target_center - scroll_center
    # Adjust to ensure fully visible.
    if abs_top - desired < top:
        desired = abs_top - top
    if abs_bottom - desired > bottom:
        desired = abs_bottom - bottom
    return max(0, int(desired))


def _capture_crop_from_device(
    device,
    bounds: Tuple[int, int, int, int],
    out_name: str,
    width: int,
    height: int,
) -> str:
    """
    Capture a full screencap and crop to bounds.
    """
    x1, y1, x2, y2 = bounds
    x1 = max(0, min(width - 1, x1))
    x2 = max(1, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(1, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid crop bounds")

    img_bytes = device.screencap()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    crop = img.crop((x1, y1, x2, y2))

    os.makedirs(os.path.join("images", "crops"), exist_ok=True)
    ts = int(time.time() * 1000)
    out_path = os.path.join("images", "crops", f"{ts}_{out_name}.png")
    crop.save(out_path)
    return out_path


def _clear_crops_folder() -> None:
    crops_dir = os.path.join("images", "crops")
    if not os.path.isdir(crops_dir):
        return
    removed = 0
    for name in os.listdir(crops_dir):
        if not name.lower().endswith(".png"):
            continue
        path = os.path.join(crops_dir, name)
        try:
            os.remove(path)
            removed += 1
        except Exception as e:
            _log(f"[PHOTO] failed to remove crop {path}: {e}")
    if removed:
        _log(f"[PHOTO] cleared {removed} crop files")


def _find_visible_photo_bounds(
    nodes: List[Dict[str, Any]],
    scroll_area: Tuple[int, int, int, int],
    expected_screen_y: int,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Find the visible photo ImageView bounds closest to the expected Y on screen.
    """
    top, bottom = scroll_area[1], scroll_area[3]
    best = None
    best_dist = None
    for n in nodes:
        if n.get("cls") != "android.widget.ImageView":
            continue
        cd = (n.get("content_desc") or "").lower()
        if "photo" not in cd:
            continue
        b = n.get("bounds")
        if not b:
            continue
        if b[1] >= bottom or b[3] <= top:
            continue
        cy = _bounds_center(b)[1]
        dist = abs(cy - expected_screen_y)
        if best_dist is None or dist < best_dist:
            best = b
            best_dist = dist
    return best


def _scan_profile_single_pass(
    device,
    width: int,
    height: int,
    max_scrolls: int = 40,
    scroll_step_px: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Single-pass slow scan: extract text + biometrics, capture photos as they appear.
    """
    ui_map = {
        "prompts": [],
        "photos": [],
        "poll": {"question": "", "options": []},
        "likes": [],
        "scroll_area": None,
        "scroll_history": [],
    }
    biometrics: Dict[str, Any] = {}
    photo_paths: List[str] = []
    seen_photo_keys: Set[int] = set()
    last_capture_abs_top: Optional[int] = None
    last_capture_height: Optional[int] = None
    skip_photo_capture_once = False

    xml = _dump_ui_xml(device)
    nodes = _parse_ui_nodes(xml)
    scroll_area = _find_scroll_area(nodes)
    if not scroll_area:
        _log("[UI] No scrollable area found in XML.")
        return {"ui_map": ui_map, "biometrics": biometrics, "photo_paths": photo_paths, "scroll_offset": 0}
    ui_map["scroll_area"] = scroll_area
    name = _extract_name_from_nodes(nodes, scroll_area)
    if name:
        biometrics["Name"] = name
        _log(f"[BIOMETRICS] Name = {name}")
    offset = 0
    did_hscroll = False
    no_move = 0
    scrolls = 0

    while True:
        # Extract biometrics visible on this screen.
        updates = _extract_biometrics_from_nodes(nodes, scroll_area)
        for k, v in updates.items():
            if k not in biometrics and v not in ("", None):
                biometrics[k] = v
                _log(f"[BIOMETRICS] {k} = {v}")

        # Horizontal biometrics scroll (once), stop when no new values appear.
        if not did_hscroll and any(k in biometrics for k in ("Age", "Gender", "Sexuality")):
            nodes = _scan_biometrics_hscroll(device, nodes, scroll_area, biometrics)
            did_hscroll = True
            skip_photo_capture_once = True

        # Update prompts/polls/likes.
        _update_ui_map_text_only(ui_map, nodes, scroll_area, offset)

        # Capture primary photo if present and new.
        photo_bounds = _find_primary_photo_bounds(nodes, scroll_area)
        if photo_bounds:
            if skip_photo_capture_once:
                _log("[PHOTO] skip capture immediately after hscroll iteration")
                skip_photo_capture_once = False
            else:
                # Micro-scroll if needed to get a full square photo.
                nodes, offset, photo_bounds = _ensure_photo_square(
                    device, width, height, scroll_area, nodes, offset, photo_bounds
                )
                scroll_area = _find_scroll_area(nodes) or scroll_area
                ui_map["scroll_area"] = scroll_area
                if photo_bounds:
                    vb_w = photo_bounds[2] - photo_bounds[0]
                    vb_h = photo_bounds[3] - photo_bounds[1]
                    is_square = abs(vb_w - vb_h) <= 12
                    if not is_square:
                        _log(f"[PHOTO] skip non-square size={vb_w}x{vb_h}")
                    else:
                        abs_bounds = (
                            photo_bounds[0],
                            photo_bounds[1] + offset,
                            photo_bounds[2],
                            photo_bounds[3] + offset,
                        )
                        abs_top = abs_bounds[1]
                        key = int(round(abs_top / 50.0)) * 50
                        min_gap = int((last_capture_height or vb_h) * 0.6)
                        if last_capture_abs_top is not None:
                            gap = abs_top - last_capture_abs_top
                            if gap <= 0 or gap < min_gap:
                                _log(
                                    f"[PHOTO] skip candidate abs_top={abs_top} gap={gap} min_gap={min_gap}"
                                )
                                photo_bounds = None
                        if photo_bounds:
                            if key in seen_photo_keys:
                                _log(f"[PHOTO] skip duplicate abs_top={abs_top} key={key}")
                            else:
                                like_bounds, like_desc = _find_like_button_in_photo(nodes, photo_bounds)
                                like_abs = None
                                if like_bounds:
                                    like_abs = (
                                        like_bounds[0],
                                        like_bounds[1] + offset,
                                        like_bounds[2],
                                        like_bounds[3] + offset,
                                    )

                                try:
                                    crop_path = _capture_crop_from_device(
                                        device,
                                        photo_bounds,
                                        f"photo_{len(ui_map['photos'])+1}",
                                        width,
                                        height,
                                    )
                                    photo_paths.append(crop_path)
                                except Exception as e:
                                    _log(f"[PHOTO] capture failed: {e}")
                                    crop_path = ""

                                photo_hash = (
                                    _compute_center_ahash_from_file(crop_path) if crop_path else None
                                )
                                like_center = _bounds_center(like_abs) if like_abs else None
                                ui_map["photos"].append(
                                    {
                                        "content_desc": "photo",
                                        "abs_bounds": abs_bounds,
                                        "abs_center_y": int((abs_bounds[1] + abs_bounds[3]) / 2),
                                        "like_bounds": like_abs,
                                        "like_desc": like_desc,
                                        "crop_path": crop_path,
                                        "hash": photo_hash,
                                        "abs_top": abs_top,
                                        "like_center": like_center,
                                    }
                                )
                                seen_photo_keys.add(key)
                                last_capture_abs_top = abs_top
                                last_capture_height = vb_h
                                _log(
                                    f"[PHOTO] captured abs_top={abs_top} abs_bounds={abs_bounds} "
                                    f"like_abs={like_abs} like_center={like_center}"
                                )
        elif skip_photo_capture_once:
            # Ensure we only skip once even if no photo was visible.
            skip_photo_capture_once = False

        # Scroll down for next screen.
        if scrolls >= max_scrolls:
            _log("[SCROLL] max_scrolls reached; stopping.")
            break
        prev_nodes = nodes
        nodes, delta = _scroll_and_capture(
            device,
            width,
            height,
            scroll_area,
            "down",
            prev_nodes,
            distance_px=scroll_step_px,
        )
        scroll_area = _find_scroll_area(nodes) or scroll_area
        ui_map["scroll_area"] = scroll_area
        offset += delta
        ui_map["scroll_history"].append(delta)
        scrolls += 1

        if abs(delta) <= 5:
            no_move += 1
        else:
            no_move = 0
        if no_move >= 2:
            _log("[SCROLL] No movement detected twice; likely bottom reached.")
            break

    _assign_like_buttons(ui_map)
    _assign_ids(ui_map)
    _log(
        f"[UI] scan done prompts={len(ui_map.get('prompts', []))} "
        f"photos={len(ui_map.get('photos', []))} poll_options={len(ui_map.get('poll', {}).get('options', []))}"
    )
    for p in ui_map.get("photos", []):
        _log(
            f"[MAP] photo id={p.get('id')} abs_top={p.get('abs_top')} like_abs={p.get('like_bounds')}"
        )
    return {
        "ui_map": ui_map,
        "biometrics": biometrics,
        "photo_paths": photo_paths,
        "scroll_offset": offset,
        "scroll_area": scroll_area,
        "nodes": nodes,
    }


def _resolve_target_from_ui_map(
    ui_map: Dict[str, Any],
    target_id: str,
) -> Dict[str, Any]:
    target_id = (target_id or "").strip()
    if not target_id:
        return {}

    if target_id.startswith("prompt_"):
        for p in ui_map.get("prompts", []):
            if p.get("id") == target_id:
                if not p.get("like_bounds"):
                    return {
                        "type": "prompt",
                        "abs_bounds": None,
                        "error": "missing_like_bounds",
                        "prompt": p.get("prompt", ""),
                        "answer": p.get("answer", ""),
                        "prompt_bounds": p.get("abs_bounds"),
                    }
                return {
                    "type": "prompt",
                    "abs_bounds": p.get("like_bounds"),
                    "prompt": p.get("prompt", ""),
                    "answer": p.get("answer", ""),
                    "prompt_bounds": p.get("abs_bounds"),
                }
    if target_id.startswith("photo_"):
        for p in ui_map.get("photos", []):
            if p.get("id") == target_id:
                if not p.get("like_bounds"):
                    return {
                        "type": "photo",
                        "abs_bounds": None,
                        "error": "missing_like_bounds",
                        "photo_bounds": p.get("abs_bounds"),
                        "photo_hash": p.get("hash"),
                    }
                return {
                    "type": "photo",
                    "abs_bounds": p.get("like_bounds"),
                    "photo_bounds": p.get("abs_bounds"),
                    "photo_hash": p.get("hash"),
                }
    if target_id.startswith("poll_"):
        for opt in ui_map.get("poll", {}).get("options", []):
            if opt.get("id") == target_id:
                return {
                    "type": "poll",
                    "abs_bounds": opt.get("abs_bounds"),
                    "option_text": opt.get("text", ""),
                }
    return {}


