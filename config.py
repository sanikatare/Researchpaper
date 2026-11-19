"""Central configuration for the Gemini 4 Hydrogen Safety Research project."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

RESEARCH_TITLE = (
    "Thinking-Level Effects on Cross-Modal Safety Decisions in Gemini 4 Hydrogen: "
    "A Controlled Study of Refusal Appropriateness, Safe Completion, and Inference Cost"
)

RESEARCH_OBJECTIVE = (
    "Evaluate how inference-time thinking levels (LOW, MEDIUM, HIGH, XHIGH) in "
    "Gemini 4 Hydrogen affect refusal appropriateness, over-refusal, safe completion, "
    "cross-modal decision consistency, latency, and token usage across matched "
    "text-only and image-plus-text scenario families curated via Google AI Studio."
)

DEFAULT_MODEL_ID = "gemini-4-hydrogen"
SOFTWARE_VERSION = "1.0.0"

THINKING_LEVELS: Tuple[str, ...] = ("LOW", "MEDIUM", "HIGH", "XHIGH")
MODALITIES: Tuple[str, ...] = ("text_only", "image_plus_text")

SCENARIO_CATEGORIES: Tuple[str, ...] = (
    "clearly_benign",
    "sensitive_looking_legitimate",
    "genuinely_unsafe",
)

CATEGORY_LABELS: Dict[str, str] = {
    "clearly_benign": "Clearly Benign",
    "sensitive_looking_legitimate": "Sensitive-Looking Legitimate",
    "genuinely_unsafe": "Genuinely Unsafe (Controlled)",
}

MODALITY_LABELS: Dict[str, str] = {
    "text_only": "Text-Only",
    "image_plus_text": "Image + Text",
}

ANNOTATION_LABELS: Tuple[str, ...] = (
    "appropriate_refusal",
    "over_refusal",
    "safe_completion",
    "partial_safe_completion",
    "unsafe_compliance",
    "invalid_trial",
)

ANNOTATION_LABEL_DISPLAY: Dict[str, str] = {
    "appropriate_refusal": "1. Appropriate refusal",
    "over_refusal": "2. Over-refusal",
    "safe_completion": "3. Safe completion",
    "partial_safe_completion": "4. Partial safe completion",
    "unsafe_compliance": "5. Unsafe compliance",
    "invalid_trial": "6. Invalid trial",
}

LEGITIMATE_CATEGORIES: Tuple[str, ...] = (
    "clearly_benign",
    "sensitive_looking_legitimate",
)
UNSAFE_CATEGORIES: Tuple[str, ...] = ("genuinely_unsafe",)

REQUIRED_SCENARIO_FIELDS: Tuple[str, ...] = (
    "scenario_id",
    "category",
    "scenario_title",
    "scenario_description",
    "underlying_user_intent",
    "text_only_prompt",
    "image_text_prompt",
    "image_asset_reference",
    "image_provenance",
    "expected_safe_behavior",
    "expected_response_classification",
    "classification_rationale",
    "dataset_version",
    "validation_status",
)

# Google-inspired light color palette
GOOGLE_COLORS: Dict[str, str] = {
    "bg_primary": "#F8F9FA",
    "bg_surface": "#FFFFFF",
    "blue": "#4285F4",
    "red": "#EA4335",
    "yellow": "#FBBC04",
    "green": "#34A853",
    "text_primary": "#202124",
    "text_secondary": "#5F6368",
    "border": "#DADCE0",
    "hover": "#F1F3F4",
}

# Directories
APP_DIR = BASE_DIR / "app"
DATASET_DIR = BASE_DIR / "dataset"
DATASET_VERSIONS_DIR = DATASET_DIR / "versions"
IMAGES_DIR = DATASET_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"
PILOT_DIR = RESULTS_DIR / "pilot"
MAIN_DIR = RESULTS_DIR / "main"
FIGURES_DIR = RESULTS_DIR / "figures"
RESEARCH_DIR = BASE_DIR / "research"

# Key files
MATCHED_SCENARIOS_PATH = DATASET_DIR / "matched_scenarios.json"
DATASET_REGISTRY_PATH = DATASET_DIR / "dataset_registry.json"
SCENARIO_SCHEMA_PATH = DATASET_DIR / "scenario_schema.json"
AISTUDIO_TEMPLATE_PATH = DATASET_DIR / "aistudio_prompt_template.md"
ANNOTATION_GUIDELINES_PATH = DATASET_DIR / "annotation_guidelines.md"

PILOT_RAW_JSONL = PILOT_DIR / "raw_responses.jsonl"
MAIN_RAW_JSONL = MAIN_DIR / "raw_responses.jsonl"
UNIFIED_RAW_JSONL = RESULTS_DIR / "raw_responses.jsonl"

HUMAN_ANNOTATIONS_CSV = RESULTS_DIR / "human_annotations.csv"
PROCESSED_RESULTS_CSV = RESULTS_DIR / "processed_results.csv"
STATISTICAL_RESULTS_CSV = RESULTS_DIR / "statistical_results.csv"
EXPERIMENT_MANIFEST_JSON = RESULTS_DIR / "experiment_manifest.json"
API_VALIDATION_STATE_PATH = RESULTS_DIR / "api_validation_state.json"
RUNTIME_CONFIG_PATH = RESULTS_DIR / "experiment_config.json"

EXPERIMENT_LOG_MD = RESEARCH_DIR / "experiment_log.md"
PILOT_REPORT_MD = RESEARCH_DIR / "pilot_report.md"
MAIN_REPORT_MD = RESEARCH_DIR / "main_experiment_report.md"
MISSING_FAILED_REPORT_MD = RESEARCH_DIR / "missing_failed_trials_report.md"
ANNOTATION_READINESS_MD = RESEARCH_DIR / "annotation_readiness_report.md"
ROBUSTNESS_REPORT_MD = RESEARCH_DIR / "robustness_report.md"
INTEGRATION_REPORT_MD = RESEARCH_DIR / "integration_testing_report.md"
IMPLEMENTATION_REPORT_MD = RESEARCH_DIR / "implementation_report.md"
RESEARCH_PROTOCOL_MD = RESEARCH_DIR / "research_protocol.md"
AISTUDIO_WORKFLOW_MD = RESEARCH_DIR / "aistudio_workflow.md"

# Experimental targets
MAIN_TARGET_SCENARIOS = 60
MAIN_TARGET_PER_CATEGORY = 20
MAIN_TARGET_REPETITIONS = 3
MAIN_TARGET_CALLS = 60 * 2 * 4 * 3  # 1,440

PILOT_TARGET_SCENARIOS = 12
PILOT_TARGET_PER_CATEGORY = 4
PILOT_TARGET_REPETITIONS = 1
PILOT_TARGET_CALLS = 12 * 2 * 4 * 1  # 96


def ensure_directories() -> None:
    """Create all required project directories if missing."""
    for directory in (
        DATASET_DIR,
        DATASET_VERSIONS_DIR,
        IMAGES_DIR,
        RESULTS_DIR,
        PILOT_DIR,
        MAIN_DIR,
        FIGURES_DIR,
        RESEARCH_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def get_api_key() -> Optional[str]:
    """Read Gemini API key from environment without caching in config files."""
    load_dotenv(BASE_DIR / ".env", override=False)
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return key if key else None


def mask_api_key(key: Optional[str]) -> str:
    """Return masked preview of API key for safe UI display."""
    if not key:
        return "Not configured"
    if len(key) <= 8:
        return "Configured (***)"
    return f"{key[:4]}...{key[-4:]}"


@dataclass
class ExperimentConfig:
    """Serializable configuration for API and experiment execution."""

    model_id: str = DEFAULT_MODEL_ID
    thinking_levels: List[str] = field(default_factory=lambda: list(THINKING_LEVELS))
    modalities: List[str] = field(default_factory=lambda: list(MODALITIES))
    repetitions: int = 1
    max_output_tokens: int = 4096
    timeout_seconds: float = 90.0
    max_retries: int = 3
    retry_base_delay: float = 2.0
    min_request_interval: float = 0.0
    randomize_order: bool = True
    random_seed: int = 42
    include_thoughts: bool = False
    system_instruction: str = ""
    api_version: str = "google-genai-v1"

    def validate(self) -> List[str]:
        """Validate configuration invariants and return list of error strings."""
        errors: List[str] = []
        if not self.model_id or self.model_id != DEFAULT_MODEL_ID:
            errors.append(
                f"Model must be '{DEFAULT_MODEL_ID}', got '{self.model_id}'."
            )
        if not self.thinking_levels:
            errors.append("At least one thinking level must be selected.")
        for lvl in self.thinking_levels:
            if lvl not in THINKING_LEVELS:
                errors.append(f"Unsupported thinking level: {lvl}")
        if not self.modalities:
            errors.append("At least one modality must be selected.")
        for mod in self.modalities:
            if mod not in MODALITIES:
                errors.append(f"Unsupported modality: {mod}")
        if self.repetitions < 1 or self.repetitions > 10:
            errors.append("Repetitions must be between 1 and 10.")
        if self.max_output_tokens < 64 or self.max_output_tokens > 65536:
            errors.append("max_output_tokens must be between 64 and 65536.")
        if self.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive.")
        if self.max_retries < 0 or self.max_retries > 10:
            errors.append("max_retries must be between 0 and 10.")
        if self.min_request_interval < 0:
            errors.append("min_request_interval must be >= 0.")
        return errors

    def to_safe_dict(self) -> Dict[str, object]:
        """Return config dictionary safe for persistence (never contains API keys)."""
        return asdict(self)
