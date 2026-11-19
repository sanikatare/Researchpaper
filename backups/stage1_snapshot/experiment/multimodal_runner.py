"""Multimodal runner preparing matched text-only and image-plus-text trials."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from config import MODALITIES, ExperimentConfig
from dataset.scenario_manager import check_image_valid, resolve_image_path
from experiment.hydrogen_client import TrialCallResult
from experiment.thinking_level_runner import run_at_thinking_level


def prepare_modality_payload(
    scenario: Dict[str, Any], modality: str
) -> Tuple[str, Optional[Path]]:
    """Return (prompt, image_path) for the requested modality after validation."""
    if modality not in MODALITIES:
        raise ValueError(f"Invalid modality '{modality}'. Expected {MODALITIES}.")

    if modality == "text_only":
        prompt = str(scenario.get("text_prompt", "")).strip()
        if not prompt:
            raise ValueError(
                f"Scenario {scenario.get('scenario_id')} missing text_prompt."
            )
        return prompt, None

    prompt = str(scenario.get("image_text_prompt", "")).strip()
    if not prompt:
        raise ValueError(
            f"Scenario {scenario.get('scenario_id')} missing image_text_prompt."
        )
    raw_img = scenario.get("image_path")
    ok, msg = check_image_valid(raw_img)
    if not ok:
        raise FileNotFoundError(
            f"Scenario {scenario.get('scenario_id')} image invalid: {msg}"
        )
    resolved = resolve_image_path(raw_img)
    return prompt, resolved


def run_multimodal_trial(
    scenario: Dict[str, Any],
    modality: str,
    thinking_level: str,
    config: ExperimentConfig,
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
) -> Tuple[str, Optional[str], TrialCallResult]:
    """Execute one matched modality trial at a specified thinking level."""
    prompt, image_path = prepare_modality_payload(scenario, modality)
    result = run_at_thinking_level(
        prompt=prompt,
        thinking_level=thinking_level,
        config=config,
        image_path=image_path,
        api_key=api_key,
        client=client,
    )
    return prompt, str(image_path) if image_path else None, result
