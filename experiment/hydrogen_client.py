"""Official Google GenAI Python SDK client wrapper for Gemini 4 Hydrogen."""

from __future__ import annotations

import io
import json
import logging
import mimetypes
import random
import re
import threading
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
    API_READINESS_REPORT_MD,
    API_VALIDATION_STATE_PATH,
    DEFAULT_MODEL_ID,
    RESULTS_DIR,
    THINKING_LEVELS,
    ensure_directories,
    get_api_key,
)

API_ERROR_LOG_PATH = RESULTS_DIR / "api_errors.log"
ALLOWED_MODEL_IDENTIFIERS = (DEFAULT_MODEL_ID, f"models/{DEFAULT_MODEL_ID}")

_RATE_LOCK = threading.Lock()
_LAST_REQUEST_MONOTONIC = 0.0


def _get_api_logger() -> logging.Logger:
    """Return logger writing API errors to results/api_errors.log without credentials."""
    ensure_directories()
    logger = logging.getLogger("gemini4hydrogen.api")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(API_ERROR_LOG_PATH, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


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


def normalize_model_identifier(model_id: str) -> str:
    """Validate and normalize Gemini 4 Hydrogen model identifier."""
    cleaned = model_id.strip()
    if cleaned not in ALLOWED_MODEL_IDENTIFIERS:
        raise ValueError(
            f"Invalid experimental model '{model_id}'. "
            f"Only '{DEFAULT_MODEL_ID}' (or 'models/{DEFAULT_MODEL_ID}') is permitted."
        )
    return DEFAULT_MODEL_ID


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

    # google-genai 2.33.0 validates open string 'XHIGH' via Pydantic into ThinkingLevel.XHIGH
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


ALLOWED_IMAGE_MIME_TYPES: Dict[str, str] = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    "GIF": "image/gif",
}
ALLOWED_IMAGE_EXTENSIONS: Dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB inline limit
MIN_EXPERIMENTAL_IMAGE_DIM = 32


def validate_image_asset_metadata(image_path: Path) -> Dict[str, Any]:
    """Validate local image existence, extension, MIME type, size bounds, and Pillow decode."""
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Associated image file not found: {path}")

    ext = path.suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image file extension '{ext}' for {path.name}. "
            f"Allowed: {sorted(ALLOWED_IMAGE_EXTENSIONS.keys())}"
        )

    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        raise ValueError(f"Image file is empty (0 bytes): {path}")
    if size_bytes > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image file {path.name} ({size_bytes} bytes) exceeds 20 MB inline limit."
        )

    raw_bytes = path.read_bytes()
    with Image.open(io.BytesIO(raw_bytes)) as img:
        img.verify()
        fmt = (img.format or "").upper()

    if fmt not in ALLOWED_IMAGE_MIME_TYPES:
        raise ValueError(
            f"Unsupported image container format '{fmt}' in {path.name}."
        )

    header_mime = ALLOWED_IMAGE_MIME_TYPES[fmt]
    ext_mime = ALLOWED_IMAGE_EXTENSIONS[ext]
    if header_mime != ext_mime:
        raise ValueError(
            f"Image MIME mismatch for {path.name}: extension implies {ext_mime}, "
            f"header is {header_mime}."
        )

    with Image.open(io.BytesIO(raw_bytes)) as img2:
        w, h = img2.size
        if w < MIN_EXPERIMENTAL_IMAGE_DIM or h < MIN_EXPERIMENTAL_IMAGE_DIM:
            raise ValueError(
                f"Image {path.name} dimensions ({w}x{h}) are below experimental minimum "
                f"({MIN_EXPERIMENTAL_IMAGE_DIM}x{MIN_EXPERIMENTAL_IMAGE_DIM})."
            )

    return {
        "path": str(path),
        "mime_type": header_mime,
        "format": fmt,
        "size_bytes": len(raw_bytes),
        "width": w,
        "height": h,
        "raw_bytes": raw_bytes,
    }


def _load_image_part(image_path: Path) -> types.Part:
    """Validate and encode a local image file as an inline Gemini Part."""
    meta = validate_image_asset_metadata(image_path)
    part = types.Part.from_bytes(data=meta["raw_bytes"], mime_type=meta["mime_type"])
    inline = getattr(part, "inline_data", None)
    if inline is None or not getattr(inline, "data", None):
        raise ValueError(f"Failed to attach inline image bytes for {image_path}")
    if len(inline.data) != meta["size_bytes"] or inline.mime_type != meta["mime_type"]:
        raise ValueError(f"Attached Part verification failed for {image_path}")
    return part


def build_client(
    api_key: Optional[str] = None, timeout_seconds: float = 90.0
) -> genai.Client:
    """Instantiate the official google.genai.Client using environment credentials."""
    resolved_key = (api_key or get_api_key() or "").strip()
    if not resolved_key:
        raise MissingApiKeyError(
            "GEMINI_API_KEY is not set. Configure it in environment variables or .env."
        )
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    timeout_ms = max(1000, int(timeout_seconds * 1000))
    return genai.Client(
        api_key=resolved_key,
        http_options=types.HttpOptions(timeout=timeout_ms),
    )


def verify_model_endpoint(
    model_id: str = DEFAULT_MODEL_ID,
    timeout_seconds: float = 30.0,
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Query Google AI Studio model metadata endpoint to confirm model accessibility."""
    try:
        canonical_model = normalize_model_identifier(model_id)
        active_client = client if client is not None else build_client(
            api_key=api_key, timeout_seconds=timeout_seconds
        )
        info = active_client.models.get(model=canonical_model)
        return {
            "accessible": True,
            "requested_model": model_id,
            "canonical_model": canonical_model,
            "endpoint_name": getattr(info, "name", canonical_model),
            "display_name": getattr(info, "display_name", None),
            "supported_actions": getattr(info, "supported_actions", None),
            "input_token_limit": getattr(info, "input_token_limit", None),
            "output_token_limit": getattr(info, "output_token_limit", None),
            "error": None,
        }
    except Exception as exc:
        _get_api_logger().error("Model endpoint check failed for %s: %s", model_id, exc)
        return {
            "accessible": False,
            "requested_model": model_id,
            "canonical_model": DEFAULT_MODEL_ID,
            "endpoint_name": None,
            "display_name": None,
            "supported_actions": None,
            "input_token_limit": None,
            "output_token_limit": None,
            "error": str(exc),
        }


def _apply_rate_limit_pacing(min_interval_seconds: float) -> None:
    """Enforce minimum spacing between outbound API requests across threads."""
    global _LAST_REQUEST_MONOTONIC
    if min_interval_seconds <= 0:
        return
    with _RATE_LOCK:
        now = time.monotonic()
        elapsed = now - _LAST_REQUEST_MONOTONIC
        if _LAST_REQUEST_MONOTONIC > 0 and elapsed < min_interval_seconds:
            time.sleep(min_interval_seconds - elapsed)
        _LAST_REQUEST_MONOTONIC = time.monotonic()


def _extract_retry_after_seconds(exc: Exception) -> Optional[float]:
    """Parse optional Retry-After / retryDelay seconds hint from rate-limit errors."""
    msg = str(exc)
    match = re.search(r"retry\s*(?:after|in|delay)\D{0,10}(\d+(?:\.\d+)?)\s*s", msg, re.I)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


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
    min_request_interval: float = 0.0,
    include_thoughts: bool = False,
    system_instruction: str = "",
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
) -> TrialCallResult:
    """Execute one controlled API call to Gemini 4 Hydrogen with retries, pacing & logging."""
    modality = "image_plus_text" if image_path is not None else "text_only"
    normalized_level = thinking_level.strip().upper()
    sdk_ver = get_sdk_version()
    timestamp = datetime.now(timezone.utc).isoformat()
    logger = _get_api_logger()

    try:
        canonical_model = normalize_model_identifier(model_id)
    except ValueError as exc:
        logger.error("Invalid model identifier '%s': %s", model_id, exc)
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
            error_code="INVALID_MODEL",
            retry_count=0,
            timestamp_utc=timestamp,
            sdk_version=sdk_ver,
        )

    if max_output_tokens < 1 or max_output_tokens > 65536:
        msg = f"Invalid max_output_tokens={max_output_tokens}; must be 1..65536."
        logger.error(msg)
        return TrialCallResult(
            model_id=canonical_model,
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
            error_message=msg,
            error_code="INVALID_MAX_OUTPUT_TOKENS",
            retry_count=0,
            timestamp_utc=timestamp,
            sdk_version=sdk_ver,
        )

    try:
        thinking_cfg = resolve_thinking_config(
            normalized_level, include_thoughts=include_thoughts
        )
    except UnsupportedThinkingLevelError as exc:
        logger.error("Unsupported thinking level '%s': %s", thinking_level, exc)
        return TrialCallResult(
            model_id=canonical_model,
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
        logger.error("Input prep error (%s, %s): %s", modality, normalized_level, exc)
        return TrialCallResult(
            model_id=canonical_model,
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
        logger.error("Client initialization failed: %s", exc)
        return TrialCallResult(
            model_id=canonical_model,
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
        _apply_rate_limit_pacing(min_request_interval)
        start_time = time.perf_counter()
        try:
            response = active_client.models.generate_content(
                model=canonical_model,
                contents=contents,
                config=gen_config,
            )
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            text_out, thoughts, finish_reason, block_reason = _extract_response_fields(
                response
            )
            in_tok, out_tok, think_tok, tot_tok = _extract_usage_tokens(response)
            actual_ver = getattr(response, "model_version", None) or canonical_model
            resp_id = getattr(response, "response_id", None)

            status = "success"
            if block_reason and not text_out.strip():
                status = "blocked"

            return TrialCallResult(
                model_id=canonical_model,
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
            logger.error(
                "API call error [model=%s level=%s modality=%s attempt=%d/%d code=%s]: %s",
                canonical_model,
                normalized_level,
                modality,
                attempt + 1,
                max_retries + 1,
                err_code,
                last_error_msg,
            )
            if retryable and attempt < max_retries:
                retries_used += 1
                hinted = _extract_retry_after_seconds(exc)
                delay = (
                    hinted
                    if hinted is not None
                    else retry_base_delay * (2**attempt) + random.uniform(0.1, 0.75)
                )
                time.sleep(delay)
                continue
            return TrialCallResult(
                model_id=canonical_model,
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
        model_id=canonical_model,
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


def _default_validation_state() -> Dict[str, Any]:
    return {
        "model_id": DEFAULT_MODEL_ID,
        "sdk_version": get_sdk_version(),
        "levels": {
            lvl: {
                "validated": False,
                "supported_by_endpoint": None,
                "modality_tested": None,
                "last_tested_utc": None,
                "error": None,
            }
            for lvl in THINKING_LEVELS
        },
        "multimodal_probe": {
            "validated": False,
            "thinking_level": None,
            "last_tested_utc": None,
            "error": None,
        },
    }


def load_validation_state() -> Dict[str, Any]:
    """Load persisted thinking-level validation state (contains no credentials)."""
    ensure_directories()
    if not API_VALIDATION_STATE_PATH.exists():
        return _default_validation_state()
    try:
        loaded = json.loads(API_VALIDATION_STATE_PATH.read_text(encoding="utf-8"))
        default = _default_validation_state()
        loaded.setdefault("multimodal_probe", default["multimodal_probe"])
        levels = loaded.setdefault("levels", {})
        for lvl in THINKING_LEVELS:
            levels.setdefault(lvl, default["levels"][lvl])
            levels[lvl].setdefault("modality_tested", None)
        return loaded
    except Exception:
        return _default_validation_state()


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
    image_path: Optional[Path] = None,
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
) -> TrialCallResult:
    """Run a benign verification probe for one thinking level and record endpoint support."""
    benign_prompt = (
        "In one concise sentence, explain why laboratory glassware is rinsed with "
        "deionized water after washing."
    )
    result = call_gemini_hydrogen(
        prompt=benign_prompt,
        thinking_level=thinking_level,
        model_id=model_id,
        image_path=image_path,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        include_thoughts=False,
        api_key=api_key,
        client=client,
    )

    state = load_validation_state()
    state["model_id"] = DEFAULT_MODEL_ID
    state["sdk_version"] = result.sdk_version
    levels = state.setdefault("levels", {})
    passed = result.status == "success" and bool(result.response_text.strip())
    err_msg = (result.error_message or "").upper()
    endpoint_rejected_level = (
        not passed
        and result.error_code in ("400", "INVALID_ARGUMENT", "UNSUPPORTED_THINKING_LEVEL")
        and any(k in err_msg for k in ("THINKING", "INVALID", "UNSUPPORTED", "LEVEL"))
    )

    levels[result.thinking_level] = {
        "validated": passed,
        "supported_by_endpoint": True if passed else (False if endpoint_rejected_level else None),
        "modality_tested": result.modality,
        "last_tested_utc": result.timestamp_utc,
        "actual_model_version": result.actual_model_version,
        "latency_ms": result.latency_ms,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "thinking_tokens": result.thinking_tokens,
        "total_tokens": result.total_tokens,
        "error": result.error_message,
        "response_preview": result.response_text[:300] if result.response_text else "",
    }
    if image_path is not None:
        state["multimodal_probe"] = {
            "validated": passed,
            "thinking_level": result.thinking_level,
            "last_tested_utc": result.timestamp_utc,
            "actual_model_version": result.actual_model_version,
            "latency_ms": result.latency_ms,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "thinking_tokens": result.thinking_tokens,
            "image_asset": Path(image_path).name,
            "error": result.error_message,
        }
    save_validation_state(state)
    return result


def test_all_thinking_levels(
    model_id: str = DEFAULT_MODEL_ID,
    max_output_tokens: int = 512,
    timeout_seconds: float = 60.0,
    max_retries: int = 1,
    image_path: Optional[Path] = None,
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
            image_path=image_path,
            api_key=api_key,
            client=client,
        )
        results.append(res)
    return results


def generate_api_readiness_report(
    save_path: Path = API_READINESS_REPORT_MD,
    endpoint_check: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate an honest Markdown API configuration & thinking-level readiness report."""
    ensure_directories()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    key_present = bool(get_api_key())
    sdk_ver = get_sdk_version()
    state = load_validation_state()
    levels = state.get("levels", {})
    mm = state.get("multimodal_probe", {})

    validated_levels = [
        lvl
        for lvl in THINKING_LEVELS
        if levels.get(lvl, {}).get("validated")
        and levels.get(lvl, {}).get("supported_by_endpoint") is True
    ]
    all_four_ok = len(validated_levels) == len(THINKING_LEVELS)

    if not key_present:
        overall_status = "BLOCKED (GEMINI_API_KEY not configured in runtime environment)"
    elif not all_four_ok:
        missing = [lvl for lvl in THINKING_LEVELS if lvl not in validated_levels]
        overall_status = f"INCOMPLETE / BLOCKED (unvalidated thinking levels: {', '.join(missing)})"
    else:
        overall_status = "READY (All 4 thinking levels live-validated)"

    lines = [
        "# Gemini 4 Hydrogen API Readiness Report",
        "",
        f"**Generated:** {ts}",
        f"**Overall API Status:** `{overall_status}`",
        f"**Target Model Identifier:** `{DEFAULT_MODEL_ID}` (`models/{DEFAULT_MODEL_ID}`)",
        f"**Google GenAI SDK Version:** `{sdk_ver}`",
        f"**`GEMINI_API_KEY` Configured:** `{key_present}` (credential values never stored in reports or logs)",
        "",
        "## 1. Model Endpoint Accessibility (`models.get`)",
    ]
    if endpoint_check is not None:
        lines.extend([
            f"- **Accessible:** `{endpoint_check.get('accessible')}`",
            f"- **Endpoint Name:** `{endpoint_check.get('endpoint_name')}`",
            f"- **Display Name:** `{endpoint_check.get('display_name')}`",
            f"- **Input / Output Token Limits:** `{endpoint_check.get('input_token_limit')}` / `{endpoint_check.get('output_token_limit')}`",
            f"- **Error:** `{endpoint_check.get('error') or 'None'}`",
        ])
    else:
        lines.append("- Live `models.get` check not executed in this report generation pass.")

    lines.extend([
        "",
        "## 2. Thinking-Level Endpoint Support Matrix",
        "",
        "| Thinking Level | Validated | Endpoint Supported | Modality Tested | Latency (ms) | Input / Output / Thinking Tokens | Last Tested (UTC) | Error |",
        "|---|---|---|---|---|---|---|---|",
    ])
    for lvl in THINKING_LEVELS:
        info = levels.get(lvl, {})
        sup = info.get("supported_by_endpoint")
        sup_lbl = "SUPPORTED" if sup is True else ("UNSUPPORTED" if sup is False else "UNTESTED")
        tok_str = (
            f"{info.get('input_tokens')}/{info.get('output_tokens')}/{info.get('thinking_tokens')}"
            if any(info.get(k) is not None for k in ("input_tokens", "output_tokens", "thinking_tokens"))
            else "Unavailable"
        )
        lat_str = f"{info['latency_ms']:.1f}" if info.get("latency_ms") is not None else "N/A"
        lines.append(
            f"| `{lvl}` | `{bool(info.get('validated'))}` | `{sup_lbl}` | "
            f"`{info.get('modality_tested') or 'None'}` | {lat_str} | `{tok_str}` | "
            f"`{info.get('last_tested_utc') or 'Never'}` | {info.get('error') or 'None'} |"
        )

    lines.extend([
        "",
        "## 3. Multimodal (`image_plus_text`) Probe Status",
        f"- **Multimodal Probe Validated:** `{bool(mm.get('validated'))}`",
        f"- **Thinking Level Used:** `{mm.get('thinking_level') or 'None'}`",
        f"- **Last Tested (UTC):** `{mm.get('last_tested_utc') or 'Never'}`",
        f"- **Error:** `{mm.get('error') or 'None'}`",
        "",
        "## 4. Verification Command",
        "Run `.venv/bin/python validate_thinking_levels.py --include-multimodal` once `GEMINI_API_KEY` is set.",
    ])

    report = "\n".join(lines) + "\n"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(report, encoding="utf-8")
    return report
