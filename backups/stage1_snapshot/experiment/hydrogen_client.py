"""Official Google GenAI Python SDK client wrapper for Gemini 4 Hydrogen."""

from __future__ import annotations

import io
import json
import mimetypes
import random
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image

from config import (
    API_VALIDATION_STATE_PATH,
    DEFAULT_MODEL_ID,
    THINKING_LEVELS,
    ensure_directories,
    get_api_key,
)


class MissingApiKeyError(RuntimeError):
    """Raised when GEMINI_API_KEY is not available in the environment."""


class UnsupportedThinkingLevelError(ValueError):
    """Raised when an invalid thinking level is requested."""


@dataclass
class TrialCallResult:
    """Structured result of a single Gemini 4 Hydrogen API call."""

    model_id: str
    actual_model_version: Optional[str]
    thinking_level: str
    modality: str
    status: str  # "success" | "blocked" | "error"
    response_text: str
    thought_summary: Optional[str]
    finish_reason: Optional[str]
    block_reason: Optional[str]
    latency_ms: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    thinking_tokens: Optional[int]
    total_tokens: Optional[int]
    response_id: Optional[str]
    error_message: Optional[str]
    error_code: Optional[str]
    retry_count: int
    timestamp_utc: str
    sdk_version: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_sdk_version() -> str:
    """Return installed google-genai version string."""
    try:
        import importlib.metadata as importlib_metadata

        return importlib_metadata.version("google-genai")
    except Exception:
        return "unknown"


def resolve_thinking_config(
    thinking_level: str, include_thoughts: bool = False
) -> types.ThinkingConfig:
    """Build a ThinkingConfig for LOW, MEDIUM, HIGH, or XHIGH without simulation."""
    normalized = thinking_level.strip().upper()
    if normalized not in THINKING_LEVELS:
        raise UnsupportedThinkingLevelError(
            f"Unsupported thinking level '{thinking_level}'. "
            f"Expected one of: {', '.join(THINKING_LEVELS)}"
        )

    if hasattr(types.ThinkingLevel, normalized):
        level_value: Any = getattr(types.ThinkingLevel, normalized)
        return types.ThinkingConfig(
            thinking_level=level_value,
            include_thoughts=include_thoughts,
        )

    # google-genai 2.33.0 supports open string enum 'XHIGH' via Pydantic validation
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*is not a valid ThinkingLevel",
            category=UserWarning,
        )
        return types.ThinkingConfig(
            thinking_level=normalized,
            include_thoughts=include_thoughts,
        )


def _load_image_part(image_path: Path) -> types.Part:
    """Validate and encode a local image file as an inline Gemini Part."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Associated image file not found: {path}")

    raw_bytes = path.read_bytes()
    # Verify image integrity with Pillow
    with Image.open(io.BytesIO(raw_bytes)) as img:
        img.verify()
        fmt = (img.format or "").upper()

    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type:
        mime_map = {
            "PNG": "image/png",
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "WEBP": "image/webp",
            "GIF": "image/gif",
        }
        mime_type = mime_map.get(fmt, "image/png")

    return types.Part.from_bytes(data=raw_bytes, mime_type=mime_type)


def build_client(
    api_key: Optional[str] = None, timeout_seconds: float = 90.0
) -> genai.Client:
    """Instantiate the official google.genai.Client using environment credentials."""
    resolved_key = (api_key or get_api_key() or "").strip()
    if not resolved_key:
        raise MissingApiKeyError(
            "GEMINI_API_KEY is not set. Configure it in environment variables or .env."
        )
    timeout_ms = max(1000, int(timeout_seconds * 1000))
    return genai.Client(
        api_key=resolved_key,
        http_options=types.HttpOptions(timeout=timeout_ms),
    )


def _is_retryable_exception(exc: Exception) -> Tuple[bool, Optional[str]]:
    """Classify whether an exception is transient and eligible for controlled retry."""
    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        code_str = str(code) if code is not None else "API_ERROR"
        if code in (429, 500, 502, 503, 504):
            return True, code_str
        msg = str(exc).upper()
        if any(
            token in msg
            for token in (
                "RESOURCE_EXHAUSTED",
                "RATE_LIMIT",
                "UNAVAILABLE",
                "DEADLINE_EXCEEDED",
                "INTERNAL",
                "TIMEOUT",
            )
        ):
            return True, code_str
        return False, code_str

    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True, type(exc).__name__

    msg = str(exc).upper()
    if any(k in msg for k in ("TIMED OUT", "TIMEOUT", "CONNECTION", "429", "503")):
        return True, type(exc).__name__
    return False, type(exc).__name__


def _extract_response_fields(
    response: Any,
) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Extract response text, thoughts, finish_reason, and block_reason safely."""
    text_out = ""
    thought_parts: List[str] = []
    finish_reason: Optional[str] = None
    block_reason: Optional[str] = None

    prompt_feedback = getattr(response, "prompt_feedback", None)
    if prompt_feedback is not None:
        br = getattr(prompt_feedback, "block_reason", None)
        if br is not None:
            block_reason = getattr(br, "value", str(br))

    candidates = getattr(response, "candidates", None) or []
    if candidates:
        cand = candidates[0]
        fr = getattr(cand, "finish_reason", None)
        if fr is not None:
            finish_reason = getattr(fr, "value", str(fr))
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        non_thought_texts: List[str] = []
        for part in parts:
            part_text = getattr(part, "text", None)
            if not part_text:
                continue
            if getattr(part, "thought", False):
                thought_parts.append(part_text)
            else:
                non_thought_texts.append(part_text)
        if non_thought_texts:
            text_out = "".join(non_thought_texts)

    if not text_out:
        try:
            raw_text = getattr(response, "text", None)
            if isinstance(raw_text, str):
                text_out = raw_text
        except Exception:
            pass

    thought_summary = "\n".join(thought_parts).strip() if thought_parts else None
    return text_out, thought_summary, finish_reason, block_reason


def _extract_usage_tokens(
    response: Any,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """Extract exact token counts from response.usage_metadata without estimating."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None, None, None, None

    def _to_int(val: Any) -> Optional[int]:
        if val is None:
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    return (
        _to_int(getattr(usage, "prompt_token_count", None)),
        _to_int(getattr(usage, "candidates_token_count", None)),
        _to_int(getattr(usage, "thoughts_token_count", None)),
        _to_int(getattr(usage, "total_token_count", None)),
    )


def call_gemini_hydrogen(
    prompt: str,
    thinking_level: str,
    model_id: str = DEFAULT_MODEL_ID,
    image_path: Optional[Path] = None,
    max_output_tokens: int = 4096,
    timeout_seconds: float = 90.0,
    max_retries: int = 3,
    retry_base_delay: float = 2.0,
    include_thoughts: bool = False,
    system_instruction: str = "",
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
) -> TrialCallResult:
    """Execute one controlled API call to Gemini 4 Hydrogen with retries & telemetry."""
    modality = "image_plus_text" if image_path is not None else "text_only"
    normalized_level = thinking_level.strip().upper()
    sdk_ver = get_sdk_version()
    timestamp = datetime.now(timezone.utc).isoformat()

    if model_id != DEFAULT_MODEL_ID:
        return TrialCallResult(
            model_id=model_id,
            actual_model_version=None,
            thinking_level=normalized_level,
            modality=modality,
            status="error",
            response_text="",
            thought_summary=None,
            finish_reason=None,
            block_reason=None,
            latency_ms=0.0,
            input_tokens=None,
            output_tokens=None,
            thinking_tokens=None,
            total_tokens=None,
            response_id=None,
            error_message=(
                f"Invalid experimental model '{model_id}'. Only '{DEFAULT_MODEL_ID}' is permitted."
            ),
            error_code="INVALID_MODEL",
            retry_count=0,
            timestamp_utc=timestamp,
            sdk_version=sdk_ver,
        )

    try:
        thinking_cfg = resolve_thinking_config(
            normalized_level, include_thoughts=include_thoughts
        )
    except UnsupportedThinkingLevelError as exc:
        return TrialCallResult(
            model_id=model_id,
            actual_model_version=None,
            thinking_level=normalized_level,
            modality=modality,
            status="error",
            response_text="",
            thought_summary=None,
            finish_reason=None,
            block_reason=None,
            latency_ms=0.0,
            input_tokens=None,
            output_tokens=None,
            thinking_tokens=None,
            total_tokens=None,
            response_id=None,
            error_message=str(exc),
            error_code="UNSUPPORTED_THINKING_LEVEL",
            retry_count=0,
            timestamp_utc=timestamp,
            sdk_version=sdk_ver,
        )

    try:
        if image_path is not None:
            image_part = _load_image_part(image_path)
            contents: Any = [image_part, types.Part.from_text(text=prompt)]
        else:
            contents = prompt
    except Exception as exc:
        return TrialCallResult(
            model_id=model_id,
            actual_model_version=None,
            thinking_level=normalized_level,
            modality=modality,
            status="error",
            response_text="",
            thought_summary=None,
            finish_reason=None,
            block_reason=None,
            latency_ms=0.0,
            input_tokens=None,
            output_tokens=None,
            thinking_tokens=None,
            total_tokens=None,
            response_id=None,
            error_message=f"Failed to prepare multimodal input: {exc}",
            error_code="INPUT_PREP_ERROR",
            retry_count=0,
            timestamp_utc=timestamp,
            sdk_version=sdk_ver,
        )

    try:
        active_client = client if client is not None else build_client(
            api_key=api_key, timeout_seconds=timeout_seconds
        )
    except Exception as exc:
        return TrialCallResult(
            model_id=model_id,
            actual_model_version=None,
            thinking_level=normalized_level,
            modality=modality,
            status="error",
            response_text="",
            thought_summary=None,
            finish_reason=None,
            block_reason=None,
            latency_ms=0.0,
            input_tokens=None,
            output_tokens=None,
            thinking_tokens=None,
            total_tokens=None,
            response_id=None,
            error_message=str(exc),
            error_code="CLIENT_INIT_ERROR",
            retry_count=0,
            timestamp_utc=timestamp,
            sdk_version=sdk_ver,
        )

    gen_config_kwargs: Dict[str, Any] = {
        "thinking_config": thinking_cfg,
        "max_output_tokens": max_output_tokens,
    }
    if system_instruction.strip():
        gen_config_kwargs["system_instruction"] = system_instruction.strip()

    gen_config = types.GenerateContentConfig(**gen_config_kwargs)

    last_error_msg: Optional[str] = None
    last_error_code: Optional[str] = None
    retries_used = 0

    for attempt in range(max_retries + 1):
        start_time = time.perf_counter()
        try:
            response = active_client.models.generate_content(
                model=model_id,
                contents=contents,
                config=gen_config,
            )
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            text_out, thoughts, finish_reason, block_reason = _extract_response_fields(
                response
            )
            in_tok, out_tok, think_tok, tot_tok = _extract_usage_tokens(response)
            actual_ver = getattr(response, "model_version", None) or model_id
            resp_id = getattr(response, "response_id", None)

            status = "success"
            if block_reason and not text_out.strip():
                status = "blocked"

            return TrialCallResult(
                model_id=model_id,
                actual_model_version=actual_ver,
                thinking_level=normalized_level,
                modality=modality,
                status=status,
                response_text=text_out,
                thought_summary=thoughts,
                finish_reason=finish_reason,
                block_reason=block_reason,
                latency_ms=elapsed_ms,
                input_tokens=in_tok,
                output_tokens=out_tok,
                thinking_tokens=think_tok,
                total_tokens=tot_tok,
                response_id=resp_id,
                error_message=None,
                error_code=None,
                retry_count=retries_used,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                sdk_version=sdk_ver,
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            retryable, err_code = _is_retryable_exception(exc)
            last_error_msg = str(exc)
            last_error_code = err_code
            if retryable and attempt < max_retries:
                retries_used += 1
                delay = retry_base_delay * (2**attempt) + random.uniform(0.1, 0.75)
                time.sleep(delay)
                continue
            return TrialCallResult(
                model_id=model_id,
                actual_model_version=None,
                thinking_level=normalized_level,
                modality=modality,
                status="error",
                response_text="",
                thought_summary=None,
                finish_reason=None,
                block_reason=None,
                latency_ms=elapsed_ms,
                input_tokens=None,
                output_tokens=None,
                thinking_tokens=None,
                total_tokens=None,
                response_id=None,
                error_message=last_error_msg,
                error_code=last_error_code,
                retry_count=retries_used,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                sdk_version=sdk_ver,
            )

    return TrialCallResult(
        model_id=model_id,
        actual_model_version=None,
        thinking_level=normalized_level,
        modality=modality,
        status="error",
        response_text="",
        thought_summary=None,
        finish_reason=None,
        block_reason=None,
        latency_ms=0.0,
        input_tokens=None,
        output_tokens=None,
        thinking_tokens=None,
        total_tokens=None,
        response_id=None,
        error_message=last_error_msg or "Unknown failure",
        error_code=last_error_code or "UNKNOWN",
        retry_count=retries_used,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        sdk_version=sdk_ver,
    )


def load_validation_state() -> Dict[str, Any]:
    """Load persisted thinking-level validation state (contains no credentials)."""
    ensure_directories()
    if not API_VALIDATION_STATE_PATH.exists():
        return {
            "model_id": DEFAULT_MODEL_ID,
            "sdk_version": get_sdk_version(),
            "levels": {
                lvl: {"validated": False, "last_tested_utc": None, "error": None}
                for lvl in THINKING_LEVELS
            },
        }
    try:
        return json.loads(API_VALIDATION_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {
            "model_id": DEFAULT_MODEL_ID,
            "sdk_version": get_sdk_version(),
            "levels": {
                lvl: {"validated": False, "last_tested_utc": None, "error": None}
                for lvl in THINKING_LEVELS
            },
        }


def save_validation_state(state: Dict[str, Any]) -> None:
    """Persist API validation state without ever writing API keys."""
    ensure_directories()
    API_VALIDATION_STATE_PATH.write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )


def is_api_fully_validated() -> bool:
    """Return True only if all four thinking levels passed a real API connection test."""
    state = load_validation_state()
    levels = state.get("levels", {})
    return all(bool(levels.get(lvl, {}).get("validated")) for lvl in THINKING_LEVELS)


def test_single_thinking_level(
    thinking_level: str,
    model_id: str = DEFAULT_MODEL_ID,
    max_output_tokens: int = 512,
    timeout_seconds: float = 60.0,
    max_retries: int = 1,
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
) -> TrialCallResult:
    """Run a benign verification probe for one thinking level and update validation state."""
    benign_prompt = (
        "In one concise sentence, explain why laboratory glassware is rinsed with "
        "deionized water after washing."
    )
    result = call_gemini_hydrogen(
        prompt=benign_prompt,
        thinking_level=thinking_level,
        model_id=model_id,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        include_thoughts=False,
        api_key=api_key,
        client=client,
    )

    state = load_validation_state()
    state["model_id"] = model_id
    state["sdk_version"] = result.sdk_version
    levels = state.setdefault("levels", {})
    levels[result.thinking_level] = {
        "validated": result.status == "success" and bool(result.response_text.strip()),
        "last_tested_utc": result.timestamp_utc,
        "actual_model_version": result.actual_model_version,
        "latency_ms": result.latency_ms,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "thinking_tokens": result.thinking_tokens,
        "error": result.error_message,
        "response_preview": result.response_text[:300] if result.response_text else "",
    }
    save_validation_state(state)
    return result


def test_all_thinking_levels(
    model_id: str = DEFAULT_MODEL_ID,
    max_output_tokens: int = 512,
    timeout_seconds: float = 60.0,
    max_retries: int = 1,
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
) -> List[TrialCallResult]:
    """Individually verify LOW, MEDIUM, HIGH, and XHIGH against Gemini 4 Hydrogen."""
    results: List[TrialCallResult] = []
    for level in THINKING_LEVELS:
        res = test_single_thinking_level(
            thinking_level=level,
            model_id=model_id,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            api_key=api_key,
            client=client,
        )
        results.append(res)
    return results
