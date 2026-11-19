"""Isolated backend helpers for exploratory Experiment Lab testing and candidate scenarios."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import (
    CANDIDATE_SCENARIOS_JSON,
    DEFAULT_MODEL_ID,
    EXPLORATORY_CHAT_JSONL,
    EXPLORATORY_UPLOADS_DIR,
    THINKING_LEVELS,
    ensure_directories,
    get_api_key,
)
from experiment.hydrogen_client import (
    ALLOWED_IMAGE_EXTENSIONS,
    MAX_IMAGE_BYTES,
    call_gemini_hydrogen,
    load_validation_state,
    validate_image_asset_metadata,
)


def check_thinking_level_availability(thinking_level: str) -> Dict[str, Any]:
    """Check whether a thinking level has been validated against the real endpoint."""
    normalized = thinking_level.strip().upper()
    if normalized not in THINKING_LEVELS:
        return {
            "level": normalized,
            "available": False,
            "supported_by_endpoint": False,
            "reason": f"'{thinking_level}' is not a valid Gemini 4 Hydrogen thinking level.",
        }

    val_state = load_validation_state()
    info = val_state.get("levels", {}).get(normalized, {})
    supported = info.get("supported_by_endpoint")
    validated = bool(info.get("validated"))

    if supported is False:
        err = info.get("error") or "The model endpoint rejected this thinking level."
        return {
            "level": normalized,
            "available": False,
            "supported_by_endpoint": False,
            "reason": (
                f"Thinking level {normalized} is not supported by {DEFAULT_MODEL_ID}: {err}"
            ),
        }

    if not validated:
        if not get_api_key():
            reason = (
                f"Thinking level {normalized} is unavailable because no GEMINI_API_KEY is "
                "configured and live endpoint verification has not been completed. "
                "Configure your API key and run 'Verify Gemini API' in Settings."
            )
        else:
            err = info.get("error")
            if err:
                reason = (
                    f"Thinking level {normalized} failed its last verification check ({err}). "
                    "Re-run 'Verify Gemini API' in Settings before using it."
                )
            else:
                reason = (
                    f"Thinking level {normalized} has not been validated against the "
                    f"{DEFAULT_MODEL_ID} endpoint yet. Run 'Verify Gemini API' in Settings first."
                )
        return {
            "level": normalized,
            "available": False,
            "supported_by_endpoint": supported,
            "reason": reason,
        }

    return {
        "level": normalized,
        "available": True,
        "supported_by_endpoint": True,
        "reason": f"Validated on {DEFAULT_MODEL_ID} ({info.get('last_tested_utc') or 'verified'}).",
    }


def save_uploaded_exploratory_image(
    file_bytes: bytes,
    filename: str,
    upload_dir: Path = EXPLORATORY_UPLOADS_DIR,
) -> Tuple[Path, Dict[str, Any]]:
    """Save and validate an uploaded image for exploratory chat without touching official datasets."""
    ensure_directories()
    upload_dir.mkdir(parents=True, exist_ok=True)

    if not file_bytes:
        raise ValueError("Uploaded image file is empty.")
    if len(file_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image '{filename}' exceeds the 20 MB maximum size limit."
        )

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"
        )

    safe_stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(filename).stem)[:48] or "upload"
    unique_name = f"{safe_stem}_{uuid.uuid4().hex[:8]}{ext}"
    dest = upload_dir / unique_name
    dest.write_bytes(file_bytes)

    try:
        meta = validate_image_asset_metadata(dest)
    except Exception:
        dest.unlink(missing_ok=True)
        raise

    meta_clean = {k: v for k, v in meta.items() if k != "raw_bytes"}
    return dest, meta_clean


def send_exploratory_prompt(
    prompt: str,
    thinking_level: str,
    input_mode: str = "text_only",
    image_path: Optional[Path] = None,
    max_output_tokens: int = 4096,
    include_thoughts: bool = False,
    system_instruction: str = "",
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute one exploratory prompt strictly separated from official pilot/main trials."""
    clean_prompt = str(prompt or "").strip()
    if not clean_prompt:
        raise ValueError("Prompt cannot be empty.")

    avail = check_thinking_level_availability(thinking_level)
    if not avail["available"]:
        raise PermissionError(avail["reason"])

    normalized_mode = input_mode.strip().lower()
    if normalized_mode not in ("text_only", "image_plus_text"):
        raise ValueError("input_mode must be 'text_only' or 'image_plus_text'.")

    effective_image: Optional[Path] = None
    if normalized_mode == "image_plus_text":
        if image_path is None:
            raise ValueError("Please attach an image when 'Image + Text' mode is selected.")
        effective_image = Path(image_path)

    from experiment.storage import load_runtime_config

    runtime_cfg = load_runtime_config()
    res = call_gemini_hydrogen(
        prompt=clean_prompt,
        thinking_level=avail["level"],
        model_id=DEFAULT_MODEL_ID,
        image_path=effective_image,
        max_output_tokens=int(max_output_tokens),
        timeout_seconds=float(runtime_cfg.timeout_seconds),
        max_retries=int(runtime_cfg.max_retries),
        retry_base_delay=float(runtime_cfg.retry_base_delay),
        min_request_interval=float(getattr(runtime_cfg, "min_request_interval", 0.0)),
        include_thoughts=include_thoughts,
        system_instruction=system_instruction,
        api_key=api_key,
        client=client,
    )

    turn: Dict[str, Any] = {
        "turn_id": f"exp-{uuid.uuid4().hex[:12]}",
        "is_exploratory": True,
        "prompt": clean_prompt,
        "thinking_level": res.thinking_level,
        "modality": res.modality,
        "image_filename": effective_image.name if effective_image else None,
        "image_path": str(effective_image) if effective_image else None,
        "status": res.status,
        "response_text": res.response_text,
        "thought_summary": res.thought_summary,
        "model_id": res.model_id,
        "actual_model_version": res.actual_model_version,
        "latency_ms": res.latency_ms,
        "input_tokens": res.input_tokens,
        "output_tokens": res.output_tokens,
        "thinking_tokens": res.thinking_tokens,
        "total_tokens": res.total_tokens,
        "block_reason": res.block_reason,
        "finish_reason": res.finish_reason,
        "error_code": res.error_code,
        "error_message": res.error_message,
        "timestamp_utc": res.timestamp_utc,
        "sdk_version": res.sdk_version,
    }

    target_log = log_path if log_path is not None else EXPLORATORY_CHAT_JSONL
    target_log.parent.mkdir(parents=True, exist_ok=True)
    with target_log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(turn, ensure_ascii=False) + "\n")

    return turn


def load_candidate_scenarios(
    candidate_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load saved exploratory candidate scenarios (never auto-approved into official datasets)."""
    target_path = candidate_path if candidate_path is not None else CANDIDATE_SCENARIOS_JSON
    if not target_path.exists():
        return []
    try:
        data = json.loads(target_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("candidates"), list):
            return data["candidates"]
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def save_candidate_scenario(
    prompt: str,
    modality: str,
    thinking_level: str,
    response_text: str = "",
    actual_model_version: Optional[str] = None,
    image_filename: Optional[str] = None,
    notes: str = "",
    candidate_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Save an exploratory prompt to a separate candidate collection without altering official datasets."""
    target_path = candidate_path if candidate_path is not None else CANDIDATE_SCENARIOS_JSON
    candidates = load_candidate_scenarios(target_path)
    record: Dict[str, Any] = {
        "candidate_id": f"CAND-{uuid.uuid4().hex[:8].upper()}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt": str(prompt or "").strip(),
        "modality": modality,
        "thinking_level": thinking_level,
        "image_filename": image_filename,
        "observed_model": actual_model_version or DEFAULT_MODEL_ID,
        "response_preview": (response_text or "")[:500],
        "notes": str(notes or "").strip(),
        "review_status": "candidate_pending_curation",
        "approved": False,
        "is_official_dataset_scenario": False,
    }
    candidates.append(record)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "collection": "exploratory_candidate_scenarios",
        "notice": "Exploratory candidates only. Not part of official pilot or main dataset versions.",
        "total_candidates": len(candidates),
        "candidates": candidates,
    }
    target_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return record
