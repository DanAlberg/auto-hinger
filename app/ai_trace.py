import json
import os
from datetime import datetime
from typing import Any, List, Optional

def _ai_trace_file() -> str:
    return os.getenv("HINGE_AI_TRACE_FILE", "")


def _ai_trace_enabled() -> bool:
    return bool(_ai_trace_file())


def _ai_trace_log(lines: List[str]) -> None:
    if not _ai_trace_enabled():
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    out_lines = [f"[{ts}] {line}" for line in lines]
    try:
        with open(_ai_trace_file(), "a", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
    except Exception:
        pass


def _ai_trace_prompt_lines(prompt: str) -> List[str]:
    return ["PROMPT=<<<BEGIN", *prompt.splitlines(), "<<<END"]


def _ai_trace_image_lines(image_paths: List[str]) -> List[str]:
    lines: List[str] = []
    for p in image_paths or []:
        if not p:
            continue
        path = os.path.abspath(str(p))
        try:
            sz = os.path.getsize(path)
        except Exception:
            sz = "?"
        lines.append(f"IMAGE image_path={path} image_size={sz} bytes")
    return lines


def _ai_trace_log_response(
    call_id: str,
    model: str,
    raw: str,
    parsed: Any = None,
    duration_ms: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    lines: List[str] = []
    header = f"AI_RESP call_id={call_id} model={model}"
    if duration_ms is not None:
        header += f" duration_ms={duration_ms}"
    lines.append(header)
    if error:
        lines.append(f"ERROR={error}")
    if parsed is not None:
        try:
            lines.extend(
                ["OUTPUT=<<<BEGIN_JSON", *json.dumps(parsed, ensure_ascii=False, indent=2).splitlines(), "<<<END_JSON"]
            )
        except Exception:
            lines.extend(["OUTPUT=<<<BEGIN_TEXT", str(parsed), "<<<END_TEXT"])
    else:
        lines.extend(["OUTPUT=<<<BEGIN_TEXT", *(str(raw) or "").splitlines(), "<<<END_TEXT"])
    _ai_trace_log(lines)


