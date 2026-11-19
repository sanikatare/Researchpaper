"""End-to-end integration tests: pipeline, stats, mixed-effects, power, robustness, manifest, traceability, and 10-page Streamlit UI."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from config import ExperimentConfig
from dataset import aistudio_workflow, scenario_manager
from evaluation import annotation, metrics, robustness, statistical_analysis
from experiment import experiment_runner, hydrogen_client, manifest, storage


class _RealisticMockModels:
    def __init__(self, fail_first: bool = False) -> None:
        self.calls = []
        self.fail_first = fail_first

    def get(self, model):
        return SimpleNamespace(
            name=f"models/{model}",
            display_name="Gemini 4 Hydrogen",
            supported_actions=["generateContent"],
            input_token_limit=1000000,
            output_token_limit=65536,
        )

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.fail_first:
            self.fail_first = False
            raise TimeoutError("Transient mock timeout")
        usage = SimpleNamespace(
            prompt_token_count=55,
            candidates_token_count=30,
            thoughts_token_count=110,
            total_token_count=195,
        )
        part = SimpleNamespace(text="Structured controlled response.", thought=False)
        cand = SimpleNamespace(finish_reason="STOP", content=SimpleNamespace(parts=[part]))
        return SimpleNamespace(
            text="Structured controlled response.",
            model_version="gemini-4-hydrogen",
            response_id=f"resp-{len(self.calls)}",
            usage_metadata=usage,
            prompt_feedback=None,
            candidates=[cand],
        )


def test_full_research_pipeline_integration_and_traceability(tmp_path, monkeypatch) -> None:
    # Isolate all mutable paths in tmp_path — never mix synthetic test artifacts with results/
    monkeypatch.setattr(storage, "PILOT_DIR", tmp_path / "pilot")
    monkeypatch.setattr(storage, "MAIN_DIR", tmp_path / "main")
    monkeypatch.setattr(storage, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(storage, "EXPERIMENT_LOG_MD", tmp_path / "experiment_log.md")
    monkeypatch.setattr(manifest, "EXPERIMENT_MANIFEST_JSON", tmp_path / "experiment_manifest.json")
    monkeypatch.setattr(hydrogen_client, "API_VALIDATION_STATE_PATH", tmp_path / "api_val.json")
    monkeypatch.setattr(hydrogen_client, "API_ERROR_LOG_PATH", tmp_path / "api_errors.log")
    monkeypatch.setattr(statistical_analysis, "STATISTICAL_RESULTS_CSV", tmp_path / "statistical_results.csv")
    monkeypatch.setattr(statistical_analysis, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(robustness, "RESULTS_DIR", tmp_path)
    (tmp_path / "pilot").mkdir(parents=True, exist_ok=True)
    (tmp_path / "main").mkdir(parents=True, exist_ok=True)

    # 1. Endpoint verification & 4 thinking levels validation
    mock_client = SimpleNamespace(models=_RealisticMockModels(fail_first=True))
    ep = hydrogen_client.verify_model_endpoint(client=mock_client)
    assert ep["accessible"] is True

    monkeypatch.setattr(hydrogen_client.time, "sleep", lambda _: None)
    for lvl in ("LOW", "MEDIUM", "HIGH", "XHIGH"):
        # Retry once after initial transient failure
        res = hydrogen_client.test_single_thinking_level(lvl, max_retries=1, client=mock_client)
        assert res.status == "success"
    assert hydrogen_client.is_api_fully_validated() is True

    # 2. Load 12 real scenarios (4 per category) and execute controlled batch with pause/resume
    _, pilot_scenarios = scenario_manager.load_active_dataset("pilot")
    assert len(pilot_scenarios) == 12

    subset = pilot_scenarios[:6]  # 2 per category -> 6 scenarios
    cfg = ExperimentConfig(
        thinking_levels=["LOW", "XHIGH"],
        modalities=["text_only", "image_plus_text"],
        repetitions=2,
        max_retries=0,
        randomize_order=True,
        random_seed=42,
    )
    # 6 * 2 * 2 * 2 = 48 planned trials; pause halfway at 24
    seen = {"n": 0}

    def _pause_half() -> bool:
        return seen["n"] >= 24

    def _on_done(rec, idx, total) -> None:
        seen["n"] += 1

    run1 = experiment_runner.run_experiment_batch(
        experiment_type="pilot",
        scenarios=subset,
        config=cfg,
        should_pause=_pause_half,
        on_trial_complete=_on_done,
        client=mock_client,
    )
    assert run1["halted_reason"] == "paused"
    assert run1["executed"] == 24

    # Resume remaining 24 trials
    run2 = experiment_runner.run_experiment_batch(
        experiment_type="pilot",
        scenarios=subset,
        config=cfg,
        client=mock_client,
    )
    assert run2["skipped_existing"] == 24
    assert run2["executed"] == 24
    assert run2["halted_reason"] == "completed"

    # 3. Raw records preserved + AI Studio observation export + CSV export
    raw_recs = storage.load_trial_records("pilot")
    assert len(raw_recs) == 48
    ai_jsonl = aistudio_workflow.format_observations_for_aistudio_jsonl("pilot")
    assert len(ai_jsonl.splitlines()) == 48
    pilot_csv = tmp_path / "pilot_results.csv"
    storage.export_records_to_csv("pilot", pilot_csv)
    assert pilot_csv.exists()

    # 4. Human annotation + Kappa + adjudication + processed results merge
    ann_csv = tmp_path / "human_annotations.csv"
    monkeypatch.setattr(annotation, "load_annotations", lambda path=ann_csv: pd.read_csv(ann_csv) if ann_csv.exists() else pd.DataFrame(columns=annotation.ANNOTATION_COLUMNS))
    monkeypatch.setattr(robustness, "load_annotations", lambda path=ann_csv: pd.read_csv(ann_csv) if ann_csv.exists() else pd.DataFrame(columns=annotation.ANNOTATION_COLUMNS))
    monkeypatch.setattr(statistical_analysis, "load_annotations", lambda path=ann_csv: pd.read_csv(ann_csv) if ann_csv.exists() else pd.DataFrame(columns=annotation.ANNOTATION_COLUMNS))

    for i, r in enumerate(raw_recs):
        cat = r["category"]
        if cat == "genuinely_unsafe":
            lbl1 = "appropriate_refusal" if (r["thinking_level"] == "XHIGH" or i % 3 != 0) else "unsafe_compliance"
            lbl2 = lbl1 if i % 7 != 0 else "appropriate_refusal"
            use = None
        else:
            lbl1 = "safe_completion" if (r["modality"] == "text_only" or i % 4 != 0) else "over_refusal"
            lbl2 = lbl1
            use = 5 if lbl1 == "safe_completion" else 1
        annotation.save_annotation(r["trial_id"], "pilot", r["scenario_id"], cat, r["modality"], r["thinking_level"], r["repetition"], "annotator_1", lbl1, use, "coded 1", path=ann_csv)
        annotation.save_annotation(r["trial_id"], "pilot", r["scenario_id"], cat, r["modality"], r["thinking_level"], r["repetition"], "annotator_2", lbl2, use, "coded 2", path=ann_csv)
        if lbl1 != lbl2:
            annotation.save_annotation(r["trial_id"], "pilot", r["scenario_id"], cat, r["modality"], r["thinking_level"], r["repetition"], "adjudicator", lbl1, use, "adjudicated", path=ann_csv)

    proc_csv = tmp_path / "processed_results.csv"
    df_proc = annotation.build_processed_results("pilot", save_csv=True, annotations_path=ann_csv, output_csv=proc_csv)
    assert len(df_proc) == 48

    # 5. All 11 metrics
    core = metrics.compute_core_metrics(df_proc, group_by=["thinking_level"])
    assert len(core["definitions"]) == 11
    assert core["overall"]["evaluable_n"] == 48

    # 6. Full Statistical Analysis: GEE + Mixed-Effects + Power + McNemar + Bootstrap + PNG figures
    stats_res = statistical_analysis.run_full_statistical_analysis(df_proc, save_csv=True)
    assert (tmp_path / "statistical_results.csv").exists()
    assert (tmp_path / "statistical_mcnemar.csv").exists()
    assert (tmp_path / "statistical_bootstrap.csv").exists()
    assert (tmp_path / "statistical_power.csv").exists()
    assert not stats_res["power_assessment"].empty
    model_types = {m.get("model_type") for m in stats_res["logistic_models"]}
    assert "GEE (Exchangeable Scenario Clustering)" in model_types
    assert "Mixed-Effects Logistic (Random Scenario Intercept)" in model_types
    stat_figs = statistical_analysis.generate_statistical_figures(stats_res, tmp_path / "figs")
    assert any(p.name == "stats_power_assessment.png" for p in stat_figs)

    # 7. Full Robustness Suite (all 8 dimensions + findings + figures + report)
    rob = robustness.run_robustness_suite(df_proc, save_outputs=False)
    assert not rob["dataset_version_comparison"]["summary_table"].empty
    assert not rob["robustness_findings"].empty
    rob_figs = robustness.generate_robustness_figures(rob, tmp_path / "figs")
    assert len(rob_figs) >= 1
    rob_md = robustness.generate_robustness_report(rob, save_path=tmp_path / "robustness_report.md")
    assert "Robust vs. Sensitive Findings Summary" in rob_md

    # 8. Manifest + 100% Raw-to-Processed Traceability
    m_obj = manifest.build_experiment_manifest("pilot", save_path=tmp_path / "experiment_manifest.json", config=cfg)
    assert m_obj["execution_counts"]["pilot"]["unique_completed_conditions"] == 48
    assert m_obj["random_seed"] == 42
    trace = manifest.verify_raw_traceability(df_proc, experiment_type="pilot")
    assert trace["all_processed_traced_to_raw"] is True
    assert trace["zero_duplicate_completed_conditions"] is True


def test_streamlit_10_pages_and_google_light_theme() -> None:
    from streamlit.testing.v1 import AppTest

    from app.components import GOOGLE_CHART_SEQUENCE, GOOGLE_LIGHT_CSS
    from app.navigation import PAGES
    from config import GOOGLE_COLORS

    # Verify Google Light Theme exact palette & 10 pages
    assert len(PAGES) == 10
    palette_blob = GOOGLE_LIGHT_CSS + " " + " ".join(GOOGLE_COLORS.values()) + " " + " ".join(GOOGLE_CHART_SEQUENCE)
    for hex_code in ("#F8F9FA", "#FFFFFF", "#4285F4", "#EA4335", "#FBBC04", "#34A853", "#202124", "#5F6368", "#DADCE0"):
        assert hex_code in palette_blob

    toml_text = Path(".streamlit/config.toml").read_text(encoding="utf-8")
    assert 'base = "light"' in toml_text
    assert '#F8F9FA' in toml_text

    # Render all 10 pages cleanly through Streamlit AppTest
    app_entry = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app_entry), default_timeout=20)
    at.run()
    assert not at.exception

    page_names = list(PAGES.keys())
    for page_name in page_names:
        at.sidebar.radio[0].set_value(page_name).run()
        assert not at.exception, f"Exception on page '{page_name}': {at.exception}"
