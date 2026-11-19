"""Thinking-level specific runner for controlled reasoning-depth trials."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from config import THINKING_LEVELS, ExperimentConfig
from experiment.hydrogen_client import (
    TrialCallResult,
    UnsupportedThinkingLevelError,
    call_gemini_hydrogen,
)


def run_at_thinking_level(
    prompt: str,
    thinking_level: str,
    config: ExperimentConfig,
    image_path: Optional[Path] = None,
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
) -> TrialCallResult:
    """Execute one Gemini 4 Hydrogen request at an explicit thinking level."""
    normalized = thinking_level.strip().upper()
    if normalized not in THINKING_LEVELS:
        raise UnsupportedThinkingLevelError(
            f"Unsupported thinking level '{thinking_level}'."
        )

    return call_gemini_hydrogen(
        prompt=prompt,
        thinking_level=normalized,
        model_id=config.model_id,
        image_path=image_path,
        max_output_tokens=config.max_output_tokens,
        timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
        retry_base_delay=config.retry_base_delay,
        min_request_interval=config.min_request_interval,
        include_thoughts=config.include_thoughts,
        system_instruction=config.system_instruction,
        api_key=api_key,
        client=client,
    )
