"""Versioned dataset schema validation, AI Studio JSON/JSONL import, and approval registry."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image

from config import (
    DATASET_REGISTRY_PATH,
    DATASET_VERSIONS_DIR,
    IMAGES_DIR,
    MAIN_TARGET_PER_CATEGORY,
    MAIN_TARGET_SCENARIOS,
    MATCHED_SCENARIOS_PATH,
    PILOT_TARGET_PER_CATEGORY,
    PILOT_TARGET_SCENARIOS,
    REQUIRED_SCENARIO_FIELDS,
    SCENARIO_CATEGORIES,
    ensure_directories,
)

LAW_EXCLUSION_KEYWORDS = (
    "lawsuit",
    "legal advice",
    "legal counsel",
    "statute",
    "statutory",
    "litigation",
    "attorney",
    "courtroom",
    "subpoena",
    "felony",
    "misdemeanor",
    "jurisdiction",
    "contract law",
    "criminal law",
    "civil law",
    "lawyer",
    "paralegal",
    "verdict",
    "tort",
    "litigate",
    "prosecutor",
    "defendant",
    "indictment",
    "bail",
)

OPERATIONAL_HARM_KEYWORDS = (
    "exact chemical recipe",
    "step-by-step bomb",
    "synthesize ricin",
    "synthesize sarin",
    "synthesize vx",
    "ransomware payload",
    "zero-day exploit code",
    "lethal dose calculation",
    "how to make fentanyl",
    "improvised explosive device",
    "weaponize pathogen",
    "buffer overflow shellcode",
)

PLACEHOLDER_STRINGS = {"tbd", "todo", "n/a", "na", "none", "placeholder", "unknown", "..."}

_STOPWORDS: Set[str] = {
    "the", "and", "for", "with", "this", "that", "from", "into", "what", "how",
    "why", "when", "should", "they", "their", "these", "there", "about", "show",
    "look", "based", "using", "shown", "image", "card", "explain", "tell", "help",
    "make", "would", "could", "can", "are", "was", "were", "have", "has", "will",
    "your", "you", "our", "one", "two", "not", "but", "more", "than", "any", "all",
    "some", "also", "then", "than", "which", "where", "who",
}

_VISUAL_REF_CUES: Tuple[str, ...] = (
    "image", "picture", "diagram", "card", "shown", "figure", "photo", "label",
    "chart", "screen", "readout", "table", "slide", "placard", "ticket", "receipt",
    "poster", "log", "summary", "mockup", "layout", "schematic", "menu", "alert",
    "tracker", "notice", "bulletin", "directory", "schedule", "timetable", "guide",
    "this", "here",
)


def _content_tokens(text: str) -> Set[str]:
    return {
        w
        for w in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(w) > 2 and w not in _STOPWORDS
    }


def _is_missing_or_placeholder(val: Any) -> bool:
    if val is None:
        return True
    s = str(val).strip()
    if not s:
        return True
    return s.lower() in PLACEHOLDER_STRINGS


def normalize_scenario_record(raw: Dict[str, Any], default_version: str = "v1.0.0") -> Dict[str, Any]:
    """Map legacy key aliases non-destructively WITHOUT fabricating missing content fields."""
    s = dict(raw)
    if "text_only_prompt" not in s and "text_prompt" in s:
        s["text_only_prompt"] = s["text_prompt"]
    if "image_asset_reference" not in s and "image_path" in s:
        s["image_asset_reference"] = s["image_path"]
    if "expected_response_classification" not in s and "expected_classification" in s:
        s["expected_response_classification"] = s["expected_classification"]
    if "dataset_version" not in s or not str(s.get("dataset_version", "")).strip():
        s["dataset_version"] = default_version
    return s


def resolve_image_path(image_ref: Optional[str]) -> Optional[Path]:
    """Resolve image asset reference under dataset/images or project root."""
    if not image_ref:
        return None
    candidate = Path(str(image_ref))
    if candidate.is_absolute() and candidate.exists():
        return candidate
    in_images = IMAGES_DIR / candidate.name if len(candidate.parts) == 1 else IMAGES_DIR / candidate
    if in_images.exists():
        return in_images
    in_dataset = MATCHED_SCENARIOS_PATH.parent / candidate
    if in_dataset.exists():
        return in_dataset
    return IMAGES_DIR / candidate


def check_image_valid(image_ref: Optional[str]) -> Tuple[bool, str]:
    """Verify image exists and passes Pillow integrity decoding."""
    if not image_ref:
        return False, "Missing image_asset_reference"
    resolved = resolve_image_path(image_ref)
    if resolved is None or not resolved.exists():
        return False, f"Image file missing: {image_ref}"
    try:
        with Image.open(resolved) as img:
            img.verify()
        with Image.open(resolved) as img2:
            w, h = img2.size
        return True, f"Valid ({w}x{h})"
    except Exception as exc:
        return False, f"Invalid image ({exc})"


def check_law_exclusion(scenario: Dict[str, Any]) -> List[str]:
    """Flag prohibited law-related content across all text fields."""
    fields = (
        "scenario_title",
        "scenario_description",
        "underlying_user_intent",
        "text_only_prompt",
        "image_text_prompt",
        "expected_safe_behavior",
        "classification_rationale",
    )
    combined = " ".join(str(scenario.get(k, "")) for k in fields).lower()
    hits: List[str] = []
    for kw in LAW_EXCLUSION_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", combined):
            hits.append(kw)
    return hits


def check_operational_harm_exclusion(scenario: Dict[str, Any]) -> List[str]:
    """Flag prohibited operationally harmful payload instructions."""
    fields = (
        "scenario_title",
        "scenario_description",
        "underlying_user_intent",
        "text_only_prompt",
        "image_text_prompt",
        "expected_safe_behavior",
    )
    combined = " ".join(str(scenario.get(k, "")) for k in fields).lower()
    return [kw for kw in OPERATIONAL_HARM_KEYWORDS if kw in combined]


def check_semantic_modality_match(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Assess deterministic cross-modal semantic matching without silent rewrites."""
    txt_prompt = str(scenario.get("text_only_prompt", "") or "").strip()
    img_prompt = str(scenario.get("image_text_prompt", "") or "").strip()
    img_context = " ".join(
        [
            img_prompt,
            str(scenario.get("image_requirements", "") or ""),
            str(scenario.get("scenario_description", "") or ""),
        ]
    )

    if not txt_prompt or not img_prompt:
        return {
            "status": "FAIL",
            "overlap_ratio": 0.0,
            "reason": "One or both modality prompts are missing.",
        }
    if txt_prompt == img_prompt:
        return {
            "status": "FAIL",
            "overlap_ratio": 1.0,
            "reason": "text_only_prompt and image_text_prompt are byte-identical.",
        }

    t_txt = _content_tokens(txt_prompt)
    t_img_ctx = _content_tokens(img_context)
    t_img_only = _content_tokens(img_prompt)

    shared_ctx = t_txt & t_img_ctx
    shared_prompt = t_txt & t_img_only
    overlap_ratio = round(len(shared_ctx) / max(1, len(t_txt)), 3)

    img_lower = img_prompt.lower()
    has_visual_cue = any(re.search(rf"\b{re.escape(cue)}\b", img_lower) for cue in _VISUAL_REF_CUES)

    if len(shared_ctx) == 0 and len(shared_prompt) == 0:
        return {
            "status": "FAIL",
            "overlap_ratio": overlap_ratio,
            "reason": "No topical overlap detected between text-only prompt and image+text context.",
        }

    warnings: List[str] = []
    if not has_visual_cue:
        warnings.append("image_text_prompt lacks an explicit visual reference cue.")
    if overlap_ratio < 0.18:
        warnings.append(
            f"Low lexical overlap ({overlap_ratio:.2f}) between modalities; human semantic equivalence check recommended."
        )

    if warnings:
        return {
            "status": "NEEDS HUMAN REVIEW",
            "overlap_ratio": overlap_ratio,
            "reason": " ".join(warnings),
        }

    return {
        "status": "PASS",
        "overlap_ratio": overlap_ratio,
        "reason": f"Topical overlap verified ({overlap_ratio:.2f}) with visual reference cue.",
    }


def validate_scenario(
    raw_scenario: Dict[str, Any],
    dataset_approved: bool = False,
) -> Dict[str, Any]:
    """Run strict schema, modality, image, semantic, and exclusion checks on one scenario."""
    scenario = normalize_scenario_record(raw_scenario)
    issues: List[str] = []
    review_warnings: List[str] = []

    for req_field in REQUIRED_SCENARIO_FIELDS:
        if _is_missing_or_placeholder(scenario.get(req_field)):
            issues.append(f"Missing or placeholder value for required field: '{req_field}'.")

    sid = str(scenario.get("scenario_id", "")).strip()
    if not re.match(r"^SCN-[0-9A-Za-z_-]+$", sid):
        issues.append(f"Invalid scenario_id format '{sid}' (expected SCN-...).")

    cat = str(scenario.get("category", "")).strip()
    if cat not in SCENARIO_CATEGORIES:
        issues.append(f"Invalid category '{cat}'. Expected {SCENARIO_CATEGORIES}.")

    txt_prompt = str(scenario.get("text_only_prompt", "") or "").strip()
    img_prompt = str(scenario.get("image_text_prompt", "") or "").strip()
    if len(txt_prompt) < 10:
        issues.append("text_only_prompt too short (<10 chars).")
    if len(img_prompt) < 10:
        issues.append("image_text_prompt too short (<10 chars).")

    sem_check = check_semantic_modality_match(scenario)
    if sem_check["status"] == "FAIL":
        issues.append(f"Cross-modal semantic match failed: {sem_check['reason']}")
    elif sem_check["status"] == "NEEDS HUMAN REVIEW":
        review_warnings.append(f"Semantic match check: {sem_check['reason']}")

    img_ok, img_msg = check_image_valid(scenario.get("image_asset_reference"))
    if not img_ok:
        issues.append(img_msg)

    prov = str(scenario.get("image_provenance", "") or "").strip()
    if _is_missing_or_placeholder(prov) or len(prov) < 8:
        issues.append("Missing or insufficiently descriptive image_provenance.")

    exp_beh = str(scenario.get("expected_safe_behavior", "") or "").strip()
    if _is_missing_or_placeholder(exp_beh) or len(exp_beh) < 10:
        issues.append("Missing or insufficient expected_safe_behavior.")

    exp_cls = str(scenario.get("expected_response_classification", "") or "").strip()
    if exp_cls not in ("safe_completion", "appropriate_refusal"):
        issues.append(
            f"expected_response_classification must be 'safe_completion' or 'appropriate_refusal', got '{exp_cls}'."
        )
    else:
        if cat == "genuinely_unsafe" and exp_cls != "appropriate_refusal":
            issues.append("genuinely_unsafe scenario must expect 'appropriate_refusal'.")
        if cat in ("clearly_benign", "sensitive_looking_legitimate") and exp_cls != "safe_completion":
            issues.append("Legitimate scenario must expect 'safe_completion'.")

    rationale = str(scenario.get("classification_rationale", "") or "").strip()
    if _is_missing_or_placeholder(rationale) or len(rationale) < 10:
        issues.append("Missing or insufficient classification_rationale.")

    law_hits = check_law_exclusion(scenario)
    if law_hits:
        issues.append(f"Prohibited law-related content detected: {', '.join(law_hits)}")

    harm_hits = check_operational_harm_exclusion(scenario)
    if harm_hits:
        issues.append(f"Prohibited operational harm phrase detected: {', '.join(harm_hits)}")

    review_state = str(scenario.get("review_status", "") or "").strip().lower()
    is_human_approved = dataset_approved or review_state == "approved"
    if not is_human_approved and not issues:
        review_warnings.append("Scenario is awaiting explicit human researcher review/approval.")

    passed_technical = len(issues) == 0
    if not passed_technical:
        report_class = "FAIL"
    elif review_warnings:
        report_class = "NEEDS HUMAN REVIEW"
    else:
        report_class = "PASS"

    return {
        "scenario_id": sid,
        "technical_valid": passed_technical,
        "validation_status": "passed" if passed_technical else "failed",
        "report_classification": report_class,
        "semantic_match": sem_check,
        "image_status": img_msg,
        "issues": issues,
        "review_warnings": review_warnings,
    }


def find_duplicates(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect exact duplicates and near-duplicate scenarios across prompts & descriptions."""
    id_counts: Dict[str, int] = {}
    prompt_hashes: Dict[str, List[str]] = {}
    title_hashes: Dict[str, List[str]] = {}

    normalized = [normalize_scenario_record(raw) for raw in scenarios]
    for s in normalized:
        sid = str(s.get("scenario_id", "")).strip()
        id_counts[sid] = id_counts.get(sid, 0) + 1

        norm_prompt = re.sub(r"\s+", " ", str(s.get("text_only_prompt", "")).strip().lower())
        p_hash = hashlib.sha256(norm_prompt.encode("utf-8")).hexdigest()[:12]
        prompt_hashes.setdefault(p_hash, []).append(sid)

        norm_title = re.sub(r"\s+", " ", str(s.get("scenario_title", "")).strip().lower())
        t_hash = hashlib.sha256(norm_title.encode("utf-8")).hexdigest()[:12]
        title_hashes.setdefault(t_hash, []).append(sid)

    dup_ids = [sid for sid, cnt in id_counts.items() if cnt > 1 or not sid]
    dup_prompts = [", ".join(sids) for sids in prompt_hashes.values() if len(sids) > 1]
    dup_titles = [", ".join(sids) for sids in title_hashes.values() if len(sids) > 1]

    # Near-duplicate detection (token Jaccard >= 0.50 or SequenceMatcher >= 0.72)
    near_dups: List[Dict[str, Any]] = []
    token_cache = [_content_tokens(str(s.get("text_only_prompt", ""))) for s in normalized]
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            sid_a = str(normalized[i].get("scenario_id", "")).strip()
            sid_b = str(normalized[j].get("scenario_id", "")).strip()
            p_a = str(normalized[i].get("text_only_prompt", "")).strip().lower()
            p_b = str(normalized[j].get("text_only_prompt", "")).strip().lower()
            if not p_a or not p_b or p_a == p_b:
                continue
            ta, tb = token_cache[i], token_cache[j]
            jac = len(ta & tb) / max(1, len(ta | tb))
            seq = SequenceMatcher(None, p_a, p_b).ratio()
            if jac >= 0.50 or seq >= 0.72:
                near_dups.append(
                    {
                        "pair": f"{sid_a} <-> {sid_b}",
                        "token_jaccard": round(jac, 3),
                        "sequence_similarity": round(seq, 3),
                    }
                )

    return {
        "duplicate_ids": dup_ids,
        "duplicate_prompts": dup_prompts,
        "duplicate_titles": dup_titles,
        "near_duplicates": near_dups,
    }


def validate_dataset(
    scenarios: List[Dict[str, Any]],
    expected_scope: str = "main",
    dataset_approved: bool = False,
) -> Dict[str, Any]:
    """Validate an entire dataset version and build a PASS / FAIL / NEEDS HUMAN REVIEW report."""
    normalized = [normalize_scenario_record(s) for s in scenarios]
    per_scenario = [
        validate_scenario(s, dataset_approved=dataset_approved) for s in normalized
    ]
    dups = find_duplicates(normalized)

    # Mark near-duplicate participants as NEEDS HUMAN REVIEW unless already FAIL
    near_dup_sids: Set[str] = set()
    for nd in dups["near_duplicates"]:
        parts = [p.strip() for p in nd["pair"].split("<->")]
        near_dup_sids.update(parts)
    if near_dup_sids:
        for item in per_scenario:
            if item["scenario_id"] in near_dup_sids:
                item["review_warnings"].append(
                    "Flagged for near-duplicate prompt similarity; human review required."
                )
                if item["report_classification"] == "PASS":
                    item["report_classification"] = "NEEDS HUMAN REVIEW"

    cat_counts: Dict[str, int] = {c: 0 for c in SCENARIO_CATEGORIES}
    for s in normalized:
        c = s.get("category", "")
        if c in cat_counts:
            cat_counts[c] += 1

    scope_issues: List[str] = []
    if expected_scope == "main":
        if len(normalized) != MAIN_TARGET_SCENARIOS:
            scope_issues.append(
                f"Confirmatory main dataset requires exactly {MAIN_TARGET_SCENARIOS} scenarios (found {len(normalized)})."
            )
        for c in SCENARIO_CATEGORIES:
            if cat_counts[c] != MAIN_TARGET_PER_CATEGORY:
                scope_issues.append(
                    f"Category '{c}' requires {MAIN_TARGET_PER_CATEGORY} scenarios (found {cat_counts[c]})."
                )
    elif expected_scope == "pilot":
        if len(normalized) != PILOT_TARGET_SCENARIOS:
            scope_issues.append(
                f"Pilot dataset requires exactly {PILOT_TARGET_SCENARIOS} scenarios (found {len(normalized)})."
            )
        for c in SCENARIO_CATEGORIES:
            if cat_counts[c] != PILOT_TARGET_PER_CATEGORY:
                scope_issues.append(
                    f"Pilot category '{c}' requires {PILOT_TARGET_PER_CATEGORY} scenarios (found {cat_counts[c]})."
                )

    if dups["duplicate_ids"]:
        scope_issues.append(f"Duplicate scenario_ids: {dups['duplicate_ids']}")
    if dups["duplicate_prompts"]:
        scope_issues.append(f"Duplicate text prompts: {dups['duplicate_prompts']}")
    if dups["duplicate_titles"]:
        scope_issues.append(f"Duplicate scenario titles: {dups['duplicate_titles']}")

    tech_pass_count = sum(1 for r in per_scenario if r["technical_valid"])
    if tech_pass_count != len(normalized):
        scope_issues.append(
            f"{len(normalized) - tech_pass_count} scenario(s) failed per-item validation."
        )

    all_passed = len(scope_issues) == 0 and len(normalized) > 0

    # 12-point structured validation report matrix (PASS / FAIL / NEEDS HUMAN REVIEW)
    target_total = MAIN_TARGET_SCENARIOS if expected_scope == "main" else PILOT_TARGET_SCENARIOS
    target_per_cat = MAIN_TARGET_PER_CATEGORY if expected_scope == "main" else PILOT_TARGET_PER_CATEGORY

    missing_issues = [
        r["scenario_id"] for r in per_scenario
        if any("Missing or placeholder" in msg for msg in r["issues"])
    ]
    modality_issues = [
        r["scenario_id"] for r in per_scenario
        if any(
            "too short" in msg or "byte-identical" in msg or "modality prompts are missing" in msg
            for msg in r["issues"]
        )
    ]
    sem_fails = [r["scenario_id"] for r in per_scenario if r["semantic_match"]["status"] == "FAIL"]
    sem_reviews = [
        r["scenario_id"] for r in per_scenario if r["semantic_match"]["status"] == "NEEDS HUMAN REVIEW"
    ]
    image_fails = [
        r["scenario_id"] for r in per_scenario
        if any("Image file missing" in msg or "Invalid image" in msg or "Missing image_asset_reference" in msg for msg in r["issues"])
    ]
    prov_fails = [
        r["scenario_id"] for r in per_scenario
        if any("image_provenance" in msg for msg in r["issues"])
    ]
    behavior_fails = [
        r["scenario_id"] for r in per_scenario
        if any(
            k in msg
            for msg in r["issues"]
            for k in ("expected_safe_behavior", "expected_response_classification", "classification_rationale", "must expect")
        )
    ]
    law_fails = [
        r["scenario_id"] for r in per_scenario
        if any("law-related" in msg for msg in r["issues"])
    ]
    harm_fails = [
        r["scenario_id"] for r in per_scenario
        if any("operational harm" in msg for msg in r["issues"])
    ]

    checks_matrix = [
        {
            "check": f"Total scenario families ({target_total} expected)",
            "status": "PASS" if len(normalized) == target_total else "FAIL",
            "details": f"Found {len(normalized)} scenario families.",
        },
        {
            "check": f"Balanced categories ({target_per_cat} per category)",
            "status": "PASS" if all(cat_counts[c] == target_per_cat for c in SCENARIO_CATEGORIES) else "FAIL",
            "details": f"Counts: {cat_counts}",
        },
        {
            "check": "Unique scenario IDs",
            "status": "PASS" if not dups["duplicate_ids"] else "FAIL",
            "details": "All scenario_id values unique." if not dups["duplicate_ids"] else f"Duplicates: {dups['duplicate_ids']}",
        },
        {
            "check": "Both modalities present & distinct",
            "status": "PASS" if not modality_issues else "FAIL",
            "details": "Both text_only and image_text prompts verified." if not modality_issues else f"Issues in: {modality_issues}",
        },
        {
            "check": "Semantic matching between modalities",
            "status": "FAIL" if sem_fails else ("NEEDS HUMAN REVIEW" if sem_reviews else "PASS"),
            "details": (
                f"Failed: {sem_fails}" if sem_fails
                else (f"{len(sem_reviews)} scenarios have low lexical overlap; human semantic confirmation recommended." if sem_reviews else "All scenario pairs aligned.")
            ),
        },
        {
            "check": "Image file availability & decoding",
            "status": "PASS" if not image_fails else "FAIL",
            "details": "All referenced images present and readable." if not image_fails else f"Missing/invalid: {image_fails}",
        },
        {
            "check": "Image provenance fields",
            "status": "PASS" if not prov_fails else "FAIL",
            "details": "All scenarios include image provenance." if not prov_fails else f"Issues in: {prov_fails}",
        },
        {
            "check": "Expected behavior & classification fields",
            "status": "PASS" if not behavior_fails else "FAIL",
            "details": "Expected behavior, classification, and rationale complete & consistent." if not behavior_fails else f"Issues in: {behavior_fails}",
        },
        {
            "check": "Duplicate & near-duplicate scenarios",
            "status": (
                "FAIL" if (dups["duplicate_prompts"] or dups["duplicate_titles"])
                else ("NEEDS HUMAN REVIEW" if dups["near_duplicates"] else "PASS")
            ),
            "details": (
                f"Exact duplicates found." if (dups["duplicate_prompts"] or dups["duplicate_titles"])
                else (f"Near-duplicates flagged: {dups['near_duplicates']}" if dups["near_duplicates"] else "No duplicates or near-duplicates detected.")
            ),
        },
        {
            "check": "Missing or placeholder values",
            "status": "PASS" if not missing_issues else "FAIL",
            "details": "No missing or placeholder values." if not missing_issues else f"Missing values in: {missing_issues}",
        },
        {
            "check": "Law-related content exclusion",
            "status": "PASS" if not law_fails else "FAIL",
            "details": "Zero law-related terms detected." if not law_fails else f"Law content flagged in: {law_fails}",
        },
        {
            "check": "Unsafe operational details exclusion",
            "status": "PASS" if not harm_fails else "FAIL",
            "details": "Zero operational harm blueprints detected." if not harm_fails else f"Operational harm flagged in: {harm_fails}",
        },
        {
            "check": "Human researcher review & sign-off",
            "status": "PASS" if dataset_approved else "NEEDS HUMAN REVIEW",
            "details": "Explicitly approved by researcher." if dataset_approved else "Pending explicit human researcher review (never auto-approved).",
        },
    ]

    class_counts = {
        "PASS": sum(1 for r in per_scenario if r["report_classification"] == "PASS"),
        "FAIL": sum(1 for r in per_scenario if r["report_classification"] == "FAIL"),
        "NEEDS HUMAN REVIEW": sum(1 for r in per_scenario if r["report_classification"] == "NEEDS HUMAN REVIEW"),
    }

    if not all_passed or any(c["status"] == "FAIL" for c in checks_matrix):
        overall_class = "FAIL"
    elif any(c["status"] == "NEEDS HUMAN REVIEW" for c in checks_matrix) or class_counts["NEEDS HUMAN REVIEW"] > 0:
        overall_class = "NEEDS HUMAN REVIEW"
    else:
        overall_class = "PASS"

    return {
        "total_scenarios": len(normalized),
        "technically_valid_count": tech_pass_count,
        "category_counts": cat_counts,
        "modality_coverage": {
            "text_only_count": sum(1 for s in normalized if str(s.get("text_only_prompt", "")).strip()),
            "image_plus_text_count": sum(
                1
                for s in normalized
                if str(s.get("image_text_prompt", "")).strip()
                and check_image_valid(s.get("image_asset_reference"))[0]
            ),
        },
        "duplicates": dups,
        "scope_issues": scope_issues,
        "dataset_valid": all_passed,
        "validation_status": "valid" if all_passed else "invalid",
        "overall_classification": overall_class,
        "classification_counts": class_counts,
        "checks_matrix": checks_matrix,
        "per_scenario": per_scenario,
    }


def load_registry() -> Dict[str, Any]:
    """Load dataset version registry."""
    ensure_directories()
    if not DATASET_REGISTRY_PATH.exists():
        return {
            "active_pilot_version": None,
            "active_main_version": None,
            "versions": [],
        }
    try:
        return json.loads(DATASET_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {
            "active_pilot_version": None,
            "active_main_version": None,
            "versions": [],
        }


def save_registry(registry: Dict[str, Any]) -> None:
    """Persist dataset version registry."""
    ensure_directories()
    DATASET_REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def parse_ai_studio_dataset_bytes(raw_bytes: bytes, filename: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Parse JSON, JSONL, or CSV scenario dataset exported from Google AI Studio."""
    import csv
    import io

    text = raw_bytes.decode("utf-8").strip()
    if not text:
        raise ValueError("Uploaded dataset file is empty.")

    meta: Dict[str, Any] = {}
    ext = Path(filename).suffix.lower()

    if ext == ".csv":
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise ValueError("CSV dataset has no header row.")
        scenarios: List[Dict[str, Any]] = []
        for row in reader:
            cleaned: Dict[str, Any] = {}
            for k, v in row.items():
                if k is None:
                    continue
                key = k.strip()
                val = v.strip() if isinstance(v, str) else v
                if key == "in_pilot" and isinstance(val, str):
                    cleaned[key] = val.lower() in ("true", "1", "yes")
                elif key == "batch_number" and isinstance(val, str) and val.isdigit():
                    cleaned[key] = int(val)
                else:
                    cleaned[key] = val
            if any(str(val).strip() for val in cleaned.values()):
                scenarios.append(cleaned)
        if not scenarios:
            raise ValueError("CSV dataset contains zero scenario rows.")
        return scenarios, {"format": "csv", "columns": list(reader.fieldnames)}

    if ext == ".jsonl" or (ext != ".json" and "\n" in text and not text.startswith("{")):
        scenarios = []
        for idx, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL on line {idx}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL line {idx} must be a JSON object.")
            scenarios.append(obj)
        return scenarios, meta

    parsed = json.loads(text)
    if isinstance(parsed, list):
        return parsed, meta
    if isinstance(parsed, dict) and "scenarios" in parsed and isinstance(parsed["scenarios"], list):
        meta = {k: v for k, v in parsed.items() if k != "scenarios"}
        return parsed["scenarios"], meta
    raise ValueError(
        "JSON dataset must be a list of scenario objects or an object containing a 'scenarios' list."
    )


def import_dataset_version(
    raw_bytes: bytes,
    filename: str,
    dataset_name: str,
    dataset_version: str,
    dataset_scope: str,
    dataset_source: str = "Google AI Studio Structured Export",
    set_active_if_valid: bool = False,
) -> Dict[str, Any]:
    """Import an immutable dataset version from JSON/JSONL without modifying scenario content."""
    ensure_directories()
    ver_clean = dataset_version.strip()
    if not ver_clean:
        raise ValueError("dataset_version cannot be empty.")
    if dataset_scope not in ("pilot", "main"):
        raise ValueError("dataset_scope must be 'pilot' or 'main'.")

    registry = load_registry()
    ver_dir = DATASET_VERSIONS_DIR / ver_clean
    if any(v["dataset_version"] == ver_clean for v in registry.get("versions", [])) or (ver_dir / "scenarios.json").exists():
        raise ValueError(
            f"Dataset version '{ver_clean}' already exists. Increment the version string to preserve prior immutable versions."
        )

    scenarios_raw, file_meta = parse_ai_studio_dataset_bytes(raw_bytes, filename)
    preserved: List[Dict[str, Any]] = []
    for raw in scenarios_raw:
        # Preserve imported fields verbatim; never silently fill or alter scenario content
        item = normalize_scenario_record(raw, default_version=ver_clean)
        item["dataset_version"] = ver_clean
        item.setdefault("review_status", "pending_human_review")
        item_val = validate_scenario(item, dataset_approved=False)
        item["validation_status"] = item_val["validation_status"]
        preserved.append(item)

    validation_report = validate_dataset(preserved, expected_scope=dataset_scope, dataset_approved=False)

    ver_dir.mkdir(parents=True, exist_ok=True)
    json_path = ver_dir / "scenarios.json"
    jsonl_path = ver_dir / "scenarios.jsonl"
    report_path = ver_dir / "validation_report.json"

    payload = {
        "dataset_name": dataset_name.strip() or f"Gemini4H-{dataset_scope}-{ver_clean}",
        "dataset_version": ver_clean,
        "dataset_scope": dataset_scope,
        "dataset_source": dataset_source.strip(),
        "import_date_utc": datetime.now(timezone.utc).isoformat(),
        "source_filename": filename,
        "metadata": file_meta,
        "scenarios": preserved,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for item in preserved:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    report_path.write_text(
        json.dumps(validation_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    root_dir = MATCHED_SCENARIOS_PATH.parent.parent
    try:
        file_ref = str(json_path.relative_to(root_dir))
        jsonl_ref = str(jsonl_path.relative_to(root_dir))
    except ValueError:
        file_ref = str(json_path)
        jsonl_ref = str(jsonl_path)

    sha256 = hashlib.sha256(json_path.read_bytes()).hexdigest()
    jsonl_sha256 = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
    entry = {
        "dataset_name": payload["dataset_name"],
        "dataset_version": ver_clean,
        "dataset_scope": dataset_scope,
        "dataset_source": payload["dataset_source"],
        "import_date_utc": payload["import_date_utc"],
        "file_reference": file_ref,
        "jsonl_reference": jsonl_ref,
        "sha256": sha256,
        "jsonl_sha256": jsonl_sha256,
        "approved_sha256": None,
        "scenario_count": validation_report["total_scenarios"],
        "technically_valid_count": validation_report["technically_valid_count"],
        "category_counts": validation_report["category_counts"],
        "modality_coverage": validation_report["modality_coverage"],
        "technical_validation_status": validation_report["validation_status"],
        "overall_report_classification": validation_report["overall_classification"],
        "classification_counts": validation_report["classification_counts"],
        "scope_issues": validation_report["scope_issues"],
        # Technical validation NEVER auto-approves
        "researcher_approval_status": "pending_review",
        "reviewer_id": None,
        "approved_at_utc": None,
    }
    registry.setdefault("versions", []).append(entry)

    if set_active_if_valid and validation_report["dataset_valid"]:
        if dataset_scope == "pilot":
            registry["active_pilot_version"] = ver_clean
        else:
            registry["active_main_version"] = ver_clean

    save_registry(registry)
    return entry


def verify_version_integrity(dataset_version: str) -> Dict[str, Any]:
    """Verify immutable dataset file on disk still matches its registered SHA-256 hash."""
    entry = get_version_entry(dataset_version)
    if entry is None:
        return {"intact": False, "error": f"Version '{dataset_version}' not found in registry."}
    ver_json = DATASET_VERSIONS_DIR / dataset_version / "scenarios.json"
    if not ver_json.exists():
        return {"intact": False, "error": f"Immutable file missing: {ver_json}"}
    actual_sha = hashlib.sha256(ver_json.read_bytes()).hexdigest()
    expected_sha = entry.get("sha256")
    approved_sha = entry.get("approved_sha256")
    matches_import = actual_sha == expected_sha
    matches_approved = (approved_sha is None) or (actual_sha == approved_sha)
    return {
        "intact": matches_import and matches_approved,
        "actual_sha256": actual_sha,
        "registered_sha256": expected_sha,
        "approved_sha256": approved_sha,
        "error": None if (matches_import and matches_approved) else "SHA-256 mismatch detected on immutable dataset file.",
    }


def load_review_log(dataset_version: str) -> Dict[str, Any]:
    """Load non-mutating per-scenario human review ledger for an immutable dataset version."""
    log_path = DATASET_VERSIONS_DIR / dataset_version / "review_log.json"
    if not log_path.exists():
        return {
            "dataset_version": dataset_version,
            "scenario_reviews": {},
            "version_signoffs": [],
        }
    try:
        data = json.loads(log_path.read_text(encoding="utf-8"))
        data.setdefault("dataset_version", dataset_version)
        data.setdefault("scenario_reviews", {})
        data.setdefault("version_signoffs", [])
        return data
    except Exception:
        return {
            "dataset_version": dataset_version,
            "scenario_reviews": {},
            "version_signoffs": [],
        }


def save_review_log(dataset_version: str, review_log: Dict[str, Any]) -> Path:
    """Persist per-scenario human review ledger without modifying immutable scenarios.json."""
    ver_dir = DATASET_VERSIONS_DIR / dataset_version
    ver_dir.mkdir(parents=True, exist_ok=True)
    log_path = ver_dir / "review_log.json"
    log_path.write_text(
        json.dumps(review_log, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return log_path


def record_scenario_review(
    dataset_version: str,
    scenario_id: str,
    decision: str,
    reviewer_id: str,
    notes: str = "",
) -> Dict[str, Any]:
    """Record an individual scenario review decision with reviewer_id & timestamp (preserves dataset SHA-256)."""
    if decision not in ("pending_review", "approved", "rejected"):
        raise ValueError("decision must be 'pending_review', 'approved', or 'rejected'.")
    clean_reviewer = str(reviewer_id or "").strip()
    if not clean_reviewer:
        raise ValueError("reviewer_id is required to record a human scenario review.")

    entry = get_version_entry(dataset_version)
    if entry is None:
        raise KeyError(f"Dataset version '{dataset_version}' not found in registry.")
    scenarios = load_scenarios_for_version(dataset_version)
    if not any(str(s.get("scenario_id")) == str(scenario_id) for s in scenarios):
        raise KeyError(f"Scenario '{scenario_id}' not found in dataset version '{dataset_version}'.")

    log_data = load_review_log(dataset_version)
    ts = datetime.now(timezone.utc).isoformat()
    log_data.setdefault("scenario_reviews", {})[str(scenario_id)] = {
        "scenario_id": str(scenario_id),
        "decision": decision,
        "reviewer_id": clean_reviewer,
        "reviewed_at_utc": ts,
        "notes": str(notes or "").strip(),
    }
    save_review_log(dataset_version, log_data)

    # If a scenario is rejected while version was previously approved, automatically revoke version approval
    if decision == "rejected" and entry.get("researcher_approval_status") == "approved":
        set_version_approval(
            dataset_version,
            "pending_review",
            reviewer_id=clean_reviewer,
            notes=f"Auto-revoked after scenario {scenario_id} marked rejected.",
        )

    return log_data["scenario_reviews"][str(scenario_id)]


def set_version_approval(
    dataset_version: str,
    approval_status: str,
    reviewer_id: Optional[str] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """Explicitly record researcher approval/rejection, reviewer_id, and freeze SHA-256 for approved datasets."""
    if approval_status not in ("pending_review", "approved", "rejected"):
        raise ValueError("Invalid approval_status.")
    clean_reviewer = str(reviewer_id).strip() if reviewer_id and str(reviewer_id).strip() else None
    registry = load_registry()
    for entry in registry.get("versions", []):
        if entry["dataset_version"] == dataset_version:
            review_log = load_review_log(dataset_version)
            rejected_sids = [
                sid
                for sid, info in review_log.get("scenario_reviews", {}).items()
                if info.get("decision") == "rejected"
            ]
            if approval_status == "approved":
                if entry["technical_validation_status"] != "valid":
                    raise ValueError(
                        "Cannot approve a dataset version that failed technical validation."
                    )
                if rejected_sids:
                    raise ValueError(
                        f"Cannot approve dataset version '{dataset_version}' because scenario(s) "
                        f"{rejected_sids} were rejected during human review. Curate a new dataset version instead."
                    )
                integrity = verify_version_integrity(dataset_version)
                if not integrity["intact"]:
                    raise ValueError(integrity["error"] or "Dataset SHA-256 integrity check failed.")
                ts = datetime.now(timezone.utc).isoformat()
                entry["approved_sha256"] = integrity["actual_sha256"]
                entry["approved_at_utc"] = ts
                entry["reviewer_id"] = clean_reviewer
                entry["overall_report_classification"] = "PASS"
                review_log.setdefault("version_signoffs", []).append(
                    {
                        "approval_status": "approved",
                        "reviewer_id": clean_reviewer,
                        "timestamp_utc": ts,
                        "approved_sha256": integrity["actual_sha256"],
                        "notes": str(notes or "").strip(),
                    }
                )
                save_review_log(dataset_version, review_log)
            else:
                entry["approved_at_utc"] = None
                entry["reviewer_id"] = clean_reviewer if approval_status == "rejected" else None
                if approval_status == "pending_review":
                    entry["approved_sha256"] = None
                    entry["overall_report_classification"] = "NEEDS HUMAN REVIEW"
                else:
                    entry["overall_report_classification"] = "FAIL"
                review_log.setdefault("version_signoffs", []).append(
                    {
                        "approval_status": approval_status,
                        "reviewer_id": clean_reviewer,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "notes": str(notes or "").strip(),
                    }
                )
                save_review_log(dataset_version, review_log)
            entry["researcher_approval_status"] = approval_status
            save_registry(registry)
            return entry
    raise KeyError(f"Dataset version '{dataset_version}' not found in registry.")


def set_active_version(dataset_scope: str, dataset_version: str) -> None:
    """Select active version for pilot or main without mixing scopes."""
    if dataset_scope not in ("pilot", "main"):
        raise ValueError("dataset_scope must be 'pilot' or 'main'.")
    registry = load_registry()
    entry = next(
        (v for v in registry.get("versions", []) if v["dataset_version"] == dataset_version),
        None,
    )
    if entry is None:
        raise KeyError(f"Dataset version '{dataset_version}' not found.")
    if entry["dataset_scope"] != dataset_scope:
        raise ValueError(
            f"Version '{dataset_version}' has scope '{entry['dataset_scope']}', cannot assign to '{dataset_scope}'."
        )
    if dataset_scope == "pilot":
        registry["active_pilot_version"] = dataset_version
    else:
        registry["active_main_version"] = dataset_version
    save_registry(registry)


def get_version_entry(dataset_version: str) -> Optional[Dict[str, Any]]:
    registry = load_registry()
    return next(
        (v for v in registry.get("versions", []) if v["dataset_version"] == dataset_version),
        None,
    )


def load_scenarios_for_version(dataset_version: str) -> List[Dict[str, Any]]:
    """Load scenarios belonging to an explicit immutable dataset version."""
    ver_json = DATASET_VERSIONS_DIR / dataset_version / "scenarios.json"
    if ver_json.exists():
        data = json.loads(ver_json.read_text(encoding="utf-8"))
        return [normalize_scenario_record(s, default_version=dataset_version) for s in data.get("scenarios", [])]
    return []


def load_active_dataset(dataset_scope: str) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (registry_entry, scenarios) for active pilot or main dataset version."""
    registry = load_registry()
    ver = (
        registry.get("active_pilot_version")
        if dataset_scope == "pilot"
        else registry.get("active_main_version")
    )
    if not ver:
        return None, []
    entry = get_version_entry(ver)
    return entry, load_scenarios_for_version(ver)


def load_scenarios(path: Path = MATCHED_SCENARIOS_PATH) -> List[Dict[str, Any]]:
    """Load active main scenarios (or fallback to matched_scenarios.json)."""
    _, active_main = load_active_dataset("main")
    if active_main:
        return active_main
    ensure_directories()
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_list = data.get("scenarios", []) if isinstance(data, dict) else data
    return [normalize_scenario_record(s) for s in raw_list]


def save_scenarios(
    scenarios: List[Dict[str, Any]], path: Path = MATCHED_SCENARIOS_PATH
) -> None:
    """Persist baseline scenarios JSON."""
    ensure_directories()
    payload = {
        "schema_version": "2.0",
        "total_scenarios": len(scenarios),
        "scenarios": scenarios,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def get_pilot_scenarios(require_approved: bool = True) -> List[Dict[str, Any]]:
    """Return scenarios from the active pilot dataset version."""
    entry, scenarios = load_active_dataset("pilot")
    if entry is None:
        return []
    if require_approved and entry.get("researcher_approval_status") != "approved":
        return []
    return scenarios


def get_main_scenarios(require_approved: bool = True) -> List[Dict[str, Any]]:
    """Return scenarios from the active confirmatory main dataset version."""
    entry, scenarios = load_active_dataset("main")
    if entry is None:
        return []
    if require_approved and entry.get("researcher_approval_status") != "approved":
        return []
    return scenarios
