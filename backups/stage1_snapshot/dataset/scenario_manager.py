"""Dataset loading, technical validation, duplicate checks, and researcher review management."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from config import (
    IMAGES_DIR,
    MATCHED_SCENARIOS_PATH,
    SCENARIO_CATEGORIES,
    ensure_directories,
)

LAW_EXCLUSION_KEYWORDS = (
    "lawsuit",
    "legal advice",
    "statute",
    "litigation",
    "attorney",
    "courtroom",
    "subpoena",
    "felony",
    "jurisdiction",
    "contract law",
    "criminal law",
    "civil law",
    "lawyer",
    "paralegal",
    "verdict",
    "tort",
)


def load_scenarios(path: Path = MATCHED_SCENARIOS_PATH) -> List[Dict[str, Any]]:
    """Load matched scenario families from JSON."""
    ensure_directories()
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "scenarios" in data:
        return list(data["scenarios"])
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected dataset format in {path}")


def save_scenarios(
    scenarios: List[Dict[str, Any]], path: Path = MATCHED_SCENARIOS_PATH
) -> None:
    """Persist scenario dataset to JSON."""
    ensure_directories()
    payload = {
        "schema_version": "1.0",
        "total_scenarios": len(scenarios),
        "scenarios": scenarios,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_image_path(image_rel_or_abs: Optional[str]) -> Optional[Path]:
    """Resolve image path relative to dataset/images or project root."""
    if not image_rel_or_abs:
        return None
    candidate = Path(image_rel_or_abs)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    in_images = IMAGES_DIR / image_rel_or_abs
    if in_images.exists():
        return in_images
    in_dataset = MATCHED_SCENARIOS_PATH.parent / image_rel_or_abs
    if in_dataset.exists():
        return in_dataset
    return in_images


def check_image_valid(image_rel_or_abs: Optional[str]) -> Tuple[bool, str]:
    """Verify image exists and is readable by Pillow."""
    if not image_rel_or_abs:
        return False, "No image path specified"
    resolved = resolve_image_path(image_rel_or_abs)
    if resolved is None or not resolved.exists():
        return False, f"Image file missing: {image_rel_or_abs}"
    try:
        with Image.open(resolved) as img:
            img.verify()
        with Image.open(resolved) as img2:
            w, h = img2.size
        return True, f"OK ({w}x{h})"
    except Exception as exc:
        return False, f"Corrupt image: {exc}"


def check_law_exclusion(scenario: Dict[str, Any]) -> List[str]:
    """Flag any accidentally introduced law-related content."""
    combined = " ".join(
        str(scenario.get(k, ""))
        for k in (
            "scenario_description",
            "text_prompt",
            "image_text_prompt",
            "expected_safe_behavior",
            "image_caption",
        )
    ).lower()
    hits: List[str] = []
    for kw in LAW_EXCLUSION_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", combined):
            hits.append(kw)
    return hits


def validate_scenario(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Run technical validation on one scenario family without auto-approving it."""
    issues: List[str] = []

    sid = str(scenario.get("scenario_id", "")).strip()
    if not sid:
        issues.append("Missing scenario_id.")

    cat = str(scenario.get("category", "")).strip()
    if cat not in SCENARIO_CATEGORIES:
        issues.append(f"Invalid category '{cat}'.")

    desc = str(scenario.get("scenario_description", "")).strip()
    if len(desc) < 10:
        issues.append("scenario_description is missing or too short.")

    text_prompt = str(scenario.get("text_prompt", "")).strip()
    img_prompt = str(scenario.get("image_text_prompt", "")).strip()
    if len(text_prompt) < 10:
        issues.append("text_prompt is missing or too short.")
    if len(img_prompt) < 10:
        issues.append("image_text_prompt is missing or too short.")

    if text_prompt and img_prompt and text_prompt == img_prompt:
        issues.append(
            "text_prompt and image_text_prompt are byte-identical; image_text_prompt "
            "should naturally reference the paired image while matching intent."
        )

    img_ok, img_msg = check_image_valid(scenario.get("image_path"))
    if not img_ok:
        issues.append(img_msg)

    expected_behavior = str(scenario.get("expected_safe_behavior", "")).strip()
    if len(expected_behavior) < 10:
        issues.append("expected_safe_behavior is missing or too short.")

    expected_cls = str(scenario.get("expected_classification", "")).strip()
    valid_expected = {
        "safe_completion",
        "appropriate_refusal",
    }
    if expected_cls not in valid_expected:
        issues.append(
            f"expected_classification must be one of {sorted(valid_expected)}, got '{expected_cls}'."
        )
    else:
        if cat == "genuinely_unsafe" and expected_cls != "appropriate_refusal":
            issues.append(
                "Genuinely unsafe scenario must have expected_classification='appropriate_refusal'."
            )
        if cat in ("clearly_benign", "sensitive_looking_legitimate") and expected_cls != "safe_completion":
            issues.append(
                "Legitimate scenario must have expected_classification='safe_completion'."
            )

    law_hits = check_law_exclusion(scenario)
    if law_hits:
        issues.append(f"Excluded law-related terms found: {', '.join(law_hits)}")

    technical_passed = len(issues) == 0
    return {
        "scenario_id": sid,
        "technical_valid": technical_passed,
        "image_status": img_msg,
        "issues": issues,
    }


def find_duplicates(scenarios: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Detect duplicate scenario IDs or identical normalized prompts."""
    id_counts: Dict[str, int] = {}
    prompt_hashes: Dict[str, List[str]] = {}
    for s in scenarios:
        sid = str(s.get("scenario_id", "")).strip()
        id_counts[sid] = id_counts.get(sid, 0) + 1
        norm = re.sub(r"\s+", " ", str(s.get("text_prompt", "")).strip().lower())
        h = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12]
        prompt_hashes.setdefault(h, []).append(sid)

    dup_ids = [sid for sid, cnt in id_counts.items() if cnt > 1 or not sid]
    dup_prompts = [
        ", ".join(sids) for sids in prompt_hashes.values() if len(sids) > 1
    ]
    return {"duplicate_ids": dup_ids, "duplicate_prompts": dup_prompts}


def validate_dataset(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate entire dataset and report technical + researcher review readiness."""
    per_scenario = [validate_scenario(s) for s in scenarios]
    dups = find_duplicates(scenarios)
    cat_counts: Dict[str, int] = {c: 0 for c in SCENARIO_CATEGORIES}
    approved_counts: Dict[str, int] = {c: 0 for c in SCENARIO_CATEGORIES}
    for s in scenarios:
        c = s.get("category", "")
        if c in cat_counts:
            cat_counts[c] += 1
            if s.get("researcher_approved") is True:
                approved_counts[c] += 1

    tech_pass_count = sum(1 for r in per_scenario if r["technical_valid"])
    approved_total = sum(1 for s in scenarios if s.get("researcher_approved") is True)
    pilot_count = sum(1 for s in scenarios if s.get("in_pilot") is True)

    return {
        "total_scenarios": len(scenarios),
        "technically_valid_count": tech_pass_count,
        "researcher_approved_count": approved_total,
        "pilot_selected_count": pilot_count,
        "category_counts": cat_counts,
        "approved_by_category": approved_counts,
        "duplicates": dups,
        "per_scenario": per_scenario,
    }


def get_pilot_scenarios(
    scenarios: List[Dict[str, Any]], require_approved: bool = True
) -> List[Dict[str, Any]]:
    """Return scenarios flagged for the pilot experiment."""
    selected = [s for s in scenarios if s.get("in_pilot") is True]
    if require_approved:
        selected = [s for s in selected if s.get("researcher_approved") is True]
    return selected


def get_main_scenarios(
    scenarios: List[Dict[str, Any]], require_approved: bool = True
) -> List[Dict[str, Any]]:
    """Return scenarios eligible for the main experiment."""
    if require_approved:
        return [s for s in scenarios if s.get("researcher_approved") is True]
    return list(scenarios)
