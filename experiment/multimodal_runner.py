"""Multimodal runner preparing matched text-only and image-plus-text trials."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from config import MODALITIES, ExperimentConfig
from dataset.scenario_manager import (
    check_semantic_modality_match,
    normalize_scenario_record,
    resolve_image_path,
)
from experiment.hydrogen_client import (
    TrialCallResult,
    _load_image_part,
    validate_image_asset_metadata,
)
from experiment.thinking_level_runner import run_at_thinking_level

PLACEHOLDER_IMAGE_TOKENS = ("placeholder", "dummy", "mock_image", "stub", "tbd", "todo")


@dataclass
class MultimodalPayload:
    """Validated modality request payload and image provenance metadata."""

    prompt: str
    modality: str
    image_path: Optional[Path]
    image_asset_reference: Optional[str]
    image_mime_type: Optional[str]
    image_size_bytes: Optional[int]
    image_width: Optional[int]
    image_height: Optional[int]
    attachment_verified: bool
    dataset_version: Optional[str]
    semantic_match_status: str


def _reject_placeholder_image(scenario: Dict[str, Any], raw_img: Any, resolved: Path) -> None:
    """Reject placeholder images so real experiments never use placeholder assets."""
    ref_lower = str(raw_img or "").strip().lower()
    name_lower = resolved.name.lower()
    prov = scenario.get("image_provenance")
    prov_str = str(prov or "").lower()

    for tok in PLACEHOLDER_IMAGE_TOKENS:
        if tok in ref_lower or tok in name_lower:
            raise ValueError(
                f"Placeholder image reference rejected for scenario {scenario.get('scenario_id')}: {raw_img}"
            )
    if isinstance(prov, dict) and prov.get("is_placeholder") is True:
        raise ValueError(
            f"Image marked is_placeholder=True rejected for scenario {scenario.get('scenario_id')}."
        )
    if "placeholder" in prov_str:
        raise ValueError(
            f"Placeholder image provenance rejected for scenario {scenario.get('scenario_id')}."
        )


def prepare_modality_bundle(
    scenario: Dict[str, Any],
    modality: str,
    validate_semantic_consistency: bool = True,
) -> MultimodalPayload:
    """Validate and construct modality payload with full image & semantic verification."""
    if modality not in MODALITIES:
        raise ValueError(f"Invalid modality '{modality}'. Expected {MODALITIES}.")

    norm = normalize_scenario_record(scenario)
    sid = norm.get("scenario_id")
    dataset_version = str(norm.get("dataset_version") or "").strip() or None

    sem_status = "SKIPPED"
    if validate_semantic_consistency:
        sem = check_semantic_modality_match(norm)
        sem_status = sem["status"]
        if sem["status"] == "FAIL":
            raise ValueError(
                f"Scenario {sid} semantic modality consistency failed: {sem['reason']}"
            )

    if modality == "text_only":
        prompt = str(norm.get("text_only_prompt") or "").strip()
        if not prompt:
            raise ValueError(f"Scenario {sid} missing text_only_prompt.")
        return MultimodalPayload(
            prompt=prompt,
            modality="text_only",
            image_path=None,
            image_asset_reference=None,
            image_mime_type=None,
            image_size_bytes=None,
            image_width=None,
            image_height=None,
            attachment_verified=False,
            dataset_version=dataset_version,
            semantic_match_status=sem_status,
        )

    prompt = str(norm.get("image_text_prompt") or "").strip()
    if not prompt:
        raise ValueError(f"Scenario {sid} missing image_text_prompt.")

    raw_img = norm.get("image_asset_reference")
    if not raw_img:
        raise FileNotFoundError(f"Scenario {sid} missing image_asset_reference.")

    resolved = resolve_image_path(str(raw_img))
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(f"Scenario {sid} image file missing: {raw_img}")

    _reject_placeholder_image(norm, raw_img, resolved)
    meta = validate_image_asset_metadata(resolved)
    # Verify official Google GenAI inline Part attachment succeeds before dispatch
    _load_image_part(resolved)

    return MultimodalPayload(
        prompt=prompt,
        modality="image_plus_text",
        image_path=resolved,
        image_asset_reference=str(raw_img),
        image_mime_type=meta["mime_type"],
        image_size_bytes=meta["size_bytes"],
        image_width=meta["width"],
        image_height=meta["height"],
        attachment_verified=True,
        dataset_version=dataset_version,
        semantic_match_status=sem_status,
    )


def prepare_modality_payload(
    scenario: Dict[str, Any],
    modality: str,
    validate_semantic_consistency: bool = True,
) -> Tuple[str, Optional[Path]]:
    """Return (prompt, image_path) for the requested modality after full validation."""
    bundle = prepare_modality_bundle(
        scenario,
        modality,
        validate_semantic_consistency=validate_semantic_consistency,
    )
    return bundle.prompt, bundle.image_path


def run_multimodal_trial_with_metadata(
    scenario: Dict[str, Any],
    modality: str,
    thinking_level: str,
    config: ExperimentConfig,
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
    validate_semantic_consistency: bool = True,
) -> Tuple[MultimodalPayload, TrialCallResult]:
    """Execute one matched modality trial and return full asset metadata + API result."""
    bundle = prepare_modality_bundle(
        scenario,
        modality,
        validate_semantic_consistency=validate_semantic_consistency,
    )
    result = run_at_thinking_level(
        prompt=bundle.prompt,
        thinking_level=thinking_level,
        config=config,
        image_path=bundle.image_path,
        api_key=api_key,
        client=client,
    )
    return bundle, result


def run_multimodal_trial(
    scenario: Dict[str, Any],
    modality: str,
    thinking_level: str,
    config: ExperimentConfig,
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
) -> Tuple[str, Optional[str], TrialCallResult]:
    """Execute one matched modality trial at a specified thinking level."""
    bundle, result = run_multimodal_trial_with_metadata(
        scenario=scenario,
        modality=modality,
        thinking_level=thinking_level,
        config=config,
        api_key=api_key,
        client=client,
    )
    return (
        bundle.prompt,
        str(bundle.image_path) if bundle.image_path else None,
        result,
    )

