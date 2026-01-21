import os
from typing import Any, Dict, List, Optional, Tuple
from types import SimpleNamespace

from openai import OpenAI

try:
    from google import genai
except Exception:
    genai = None


_CLIENT_CACHE: Dict[Tuple[Any, ...], Any] = {}


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")


def get_llm_provider() -> str:
    return (os.getenv("LLM_PROVIDER") or "openai").strip().lower()


def get_default_model() -> str:
    provider = get_llm_provider()
    if provider == "gemini":
        return (os.getenv("LLM_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.5-pro").strip()
    return (os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-5").strip()


def get_default_small_model() -> str:
    provider = get_llm_provider()
    if provider == "gemini":
        return (os.getenv("LLM_SMALL_MODEL") or os.getenv("GEMINI_SMALL_MODEL") or "gemini-2.5-flash").strip()
    return (os.getenv("LLM_SMALL_MODEL") or os.getenv("OPENAI_SMALL_MODEL") or "gpt-5-mini").strip()


def resolve_model(model: Optional[str]) -> str:
    model = (model or "").strip()
    provider = get_llm_provider()
    if not model:
        return get_default_model()
    if model in ("small", "small_model"):
        return get_default_small_model()
    if model in ("large", "large_model", "default"):
        return get_default_model()
    if provider != "gemini":
        return model
    if model.startswith("gemini-"):
        return model
    small = get_default_small_model()
    large = get_default_model()
    if "mini" in model or "small" in model:
        return small
    return large


def get_llm_client() -> Any:
    provider = get_llm_provider()
    if provider == "gemini":
        return _get_gemini_client()
    return _get_openai_client()


def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    cache_key = ("openai", api_key)
    if cache_key not in _CLIENT_CACHE:
        _CLIENT_CACHE[cache_key] = OpenAI(api_key=api_key)
    return _CLIENT_CACHE[cache_key]


def _get_gemini_client() -> Any:
    if genai is None:
        raise RuntimeError("google-genai is required for GEMINI usage.")
    use_vertex = _env_bool("GEMINI_USE_VERTEX", default=False)
    project_id = (os.getenv("GEMINI_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
    location = (os.getenv("GEMINI_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION") or "").strip()
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    cache_key = ("gemini", use_vertex, project_id, location, api_key)
    if cache_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[cache_key]

    if use_vertex:
        if not project_id:
            raise RuntimeError("GEMINI_PROJECT_ID is required when GEMINI_USE_VERTEX=1.")
        if not location:
            raise RuntimeError("GEMINI_LOCATION is required when GEMINI_USE_VERTEX=1.")
        client = genai.Client(vertexai=True, project=project_id, location=location)
    else:
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required when GEMINI_USE_VERTEX=0.")
        client = genai.Client(api_key=api_key)

    compat = _GeminiOpenAICompat(client)
    _CLIENT_CACHE[cache_key] = compat
    return compat


def _split_data_url(url: str) -> Optional[Tuple[str, str]]:
    if not url.startswith("data:"):
        return None
    try:
        header, data = url.split(",", 1)
    except ValueError:
        return None
    mime = header[5:].split(";")[0].strip() if header.startswith("data:") else ""
    mime = mime or "application/octet-stream"
    return mime, data


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    texts.append(text)
        return "\n".join([t for t in texts if t]).strip()
    return ""


def _content_parts(content: Any) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    if isinstance(content, str):
        parts.append({"text": content})
        return parts
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append({"text": text})
            elif ptype == "image_url":
                image = part.get("image_url") or {}
                url = image.get("url") if isinstance(image, dict) else ""
                if not isinstance(url, str):
                    continue
                split = _split_data_url(url)
                if split:
                    mime, data = split
                    parts.append({"inline_data": {"mime_type": mime, "data": data}})
                elif url:
                    parts.append({"file_data": {"mime_type": "image/png", "file_uri": url}})
    return parts or [{"text": ""}]


def _openai_messages_to_gemini(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    contents: List[Dict[str, Any]] = []
    system_chunks: List[str] = []

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = (msg.get("role") or "user").strip().lower()
        content = msg.get("content", "")
        if role == "system":
            text = _content_text(content)
            if text:
                system_chunks.append(text)
            continue
        gemini_role = "model" if role == "assistant" else "user"
        parts = _content_parts(content)
        contents.append({"role": gemini_role, "parts": parts})

    system_instruction = "\n".join(system_chunks).strip()
    return contents, system_instruction


def _gemini_text_from_response(resp: Any) -> str:
    try:
        text = getattr(resp, "text", None)
        if isinstance(text, str) and text:
            return text
    except Exception:
        pass
    try:
        parts_out: List[str] = []
        candidates = getattr(resp, "candidates", []) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", []) if content is not None else []
            for part in parts or []:
                if isinstance(part, dict):
                    text = part.get("text")
                else:
                    text = getattr(part, "text", None)
                if isinstance(text, str):
                    parts_out.append(text)
        return "".join(parts_out)
    except Exception:
        return ""


def _wrap_openai_text(text: str) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class _GeminiChatCompletions:
    def __init__(self, parent: "_GeminiOpenAICompat") -> None:
        self._parent = parent

    def create(self, model: str, messages: List[Dict[str, Any]], response_format: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        model = resolve_model(model)
        contents, system_instruction = _openai_messages_to_gemini(messages)
        config: Dict[str, Any] = {}
        if system_instruction:
            config["system_instruction"] = system_instruction
        if isinstance(response_format, dict) and response_format.get("type") == "json_object":
            config["response_mime_type"] = "application/json"
        for k in ("temperature", "top_p", "max_output_tokens"):
            if k in kwargs:
                config[k] = kwargs[k]
        if config:
            resp = self._parent._client.models.generate_content(model=model, contents=contents, config=config)
        else:
            resp = self._parent._client.models.generate_content(model=model, contents=contents)
        text = _gemini_text_from_response(resp)
        return _wrap_openai_text(text)


class _GeminiResponses:
    def create(self, **kwargs) -> Any:
        raise NotImplementedError("Gemini provider does not support OpenAI Responses API.")


class _GeminiChat:
    def __init__(self, parent: "_GeminiOpenAICompat") -> None:
        self.completions = _GeminiChatCompletions(parent)


class _GeminiOpenAICompat:
    def __init__(self, client: Any) -> None:
        self._client = client
        self.chat = _GeminiChat(self)
        self.responses = _GeminiResponses()
