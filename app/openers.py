import json
import time
from typing import Any, Dict

from llm_client import get_default_model, get_llm_client, resolve_model
from prompts import LLM3_LONG, LLM3_SHORT, LLM4
from ai_trace import _ai_trace_log, _ai_trace_log_response, _ai_trace_prompt_lines
from runtime import _log

def run_llm3_long(extracted: Dict[str, Any], model: str | None = None) -> Dict[str, Any]:
    prompt = LLM3_LONG(extracted)
    requested_model = model or get_default_model()
    resolved_model = resolve_model(requested_model)
    trace_lines = [
        f"AI_CALL call_id=llm3_long model={resolved_model} response_format=json_object"
    ]
    trace_lines.extend(_ai_trace_prompt_lines(prompt))
    _ai_trace_log(trace_lines)
    try:
        t0 = time.perf_counter()
        resp = get_llm_client().chat.completions.create(
            model=resolved_model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        dt_ms = int((time.perf_counter() - t0) * 1000)
        raw = resp.choices[0].message.content or ""
        try:
            parsed = json.loads(raw or "{}")
        except Exception as e:
            _ai_trace_log_response(
                "llm3_long",
                resolved_model,
                raw,
                parsed=None,
                duration_ms=dt_ms,
                error=f"json_parse_error: {e}",
            )
            _log(f"[LLM3] long parse failed: {e}")
            return {}
        if not isinstance(parsed, dict):
            _ai_trace_log_response(
                "llm3_long",
                resolved_model,
                raw,
                parsed=None,
                duration_ms=dt_ms,
                error="parsed_not_dict",
            )
            return {}
        _ai_trace_log_response(
            "llm3_long",
            resolved_model,
            raw,
            parsed=parsed,
            duration_ms=dt_ms,
        )
        return parsed
    except Exception as e:
        _ai_trace_log_response(
            "llm3_long",
            resolved_model,
            raw="",
            parsed=None,
            duration_ms=None,
            error=f"call_error: {e}",
        )
        return {}


def run_llm3_short(extracted: Dict[str, Any], model: str | None = None) -> Dict[str, Any]:
    prompt = LLM3_SHORT(extracted)
    requested_model = model or get_default_model()
    resolved_model = resolve_model(requested_model)
    trace_lines = [
        f"AI_CALL call_id=llm3_short model={resolved_model} response_format=json_object"
    ]
    trace_lines.extend(_ai_trace_prompt_lines(prompt))
    _ai_trace_log(trace_lines)
    try:
        t0 = time.perf_counter()
        resp = get_llm_client().chat.completions.create(
            model=resolved_model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        dt_ms = int((time.perf_counter() - t0) * 1000)
        raw = resp.choices[0].message.content or ""
        try:
            parsed = json.loads(raw or "{}")
        except Exception as e:
            _ai_trace_log_response(
                "llm3_short",
                resolved_model,
                raw,
                parsed=None,
                duration_ms=dt_ms,
                error=f"json_parse_error: {e}",
            )
            _log(f"[LLM3] short parse failed: {e}")
            return {}
        if not isinstance(parsed, dict):
            _ai_trace_log_response(
                "llm3_short",
                resolved_model,
                raw,
                parsed=None,
                duration_ms=dt_ms,
                error="parsed_not_dict",
            )
            return {}
        _ai_trace_log_response(
            "llm3_short",
            resolved_model,
            raw,
            parsed=parsed,
            duration_ms=dt_ms,
        )
        return parsed
    except Exception as e:
        _ai_trace_log_response(
            "llm3_short",
            resolved_model,
            raw="",
            parsed=None,
            duration_ms=None,
            error=f"call_error: {e}",
        )
        return {}


def run_llm4(openers_json: Dict[str, Any], model: str | None = None) -> Dict[str, Any]:
    prompt = LLM4(openers_json)
    requested_model = model or get_default_model()
    resolved_model = resolve_model(requested_model)
    trace_lines = [
        f"AI_CALL call_id=llm4 model={resolved_model} response_format=json_object"
    ]
    trace_lines.extend(_ai_trace_prompt_lines(prompt))
    _ai_trace_log(trace_lines)
    try:
        t0 = time.perf_counter()
        resp = get_llm_client().chat.completions.create(
            model=resolved_model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        dt_ms = int((time.perf_counter() - t0) * 1000)
        raw = resp.choices[0].message.content or ""
        try:
            parsed = json.loads(raw or "{}")
        except Exception as e:
            _ai_trace_log_response(
                "llm4",
                resolved_model,
                raw,
                parsed=None,
                duration_ms=dt_ms,
                error=f"json_parse_error: {e}",
            )
            _log(f"[LLM4] parse failed: {e}")
            return {}
        try:
            _ai_trace_log_response(
                "llm4",
                resolved_model,
                raw,
                parsed=parsed,
                duration_ms=dt_ms,
            )
        except Exception:
            pass
        return parsed if isinstance(parsed, dict) else {}
    except Exception as e:
        _ai_trace_log_response(
            "llm4",
            resolved_model,
            raw="",
            parsed=None,
            duration_ms=None,
            error=f"call_error: {e}",
        )
        return {}


