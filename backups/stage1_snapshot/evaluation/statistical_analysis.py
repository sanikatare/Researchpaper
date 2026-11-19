"""Statistical inference: CIs, McNemar paired tests, effect sizes, clustered bootstrap, and GEE/mixed-effects logistic regression."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.stats.contingency_tables import mcnemar

from config import LEGITIMATE_CATEGORIES, STATISTICAL_RESULTS_CSV, THINKING_LEVELS, UNSAFE_CATEGORIES
from evaluation.metrics import compute_core_metrics, filter_evaluable_trials


def holm_bonferroni(p_values: List[Optional[float]]) -> List[Optional[float]]:
    """Apply Holm-Bonferroni step-down correction preserving None entries."""
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None and not np.isnan(p)]
    if not indexed:
        return [None] * len(p_values)
    indexed.sort(key=lambda x: x[1])
    m = len(indexed)
    adjusted: Dict[int, float] = {}
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
    return [adjusted.get(i, None) for i in range(len(p_values))]


def run_mcnemar_comparison(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    join_cols: List[str],
    outcome_col: str,
    comparison_name: str,
) -> Dict[str, Any]:
    """Run exact/asymptotic McNemar test on paired binary outcomes."""
    a_sub = df_a[join_cols + [outcome_col]].rename(columns={outcome_col: "out_a"})
    b_sub = df_b[join_cols + [outcome_col]].rename(columns={outcome_col: "out_b"})
    paired = a_sub.merge(b_sub, on=join_cols, how="inner")
    n = len(paired)
    if n < 2:
        return {
            "comparison": comparison_name,
            "outcome": outcome_col,
            "n_pairs": n,
            "b_10": None,
            "c_01": None,
            "p_value": None,
            "effect_size_cohens_g": None,
            "odds_ratio": None,
            "status": "insufficient_data",
            "note": "Need >= 2 matched pairs.",
        }

    a_val = paired["out_a"].astype(bool)
    b_val = paired["out_b"].astype(bool)

    n11 = int((a_val & b_val).sum())
    n10 = int((a_val & ~b_val).sum())
    n01 = int((~a_val & b_val).sum())
    n00 = int((~a_val & ~b_val).sum())

    discordant = n10 + n01
    if discordant == 0:
        return {
            "comparison": comparison_name,
            "outcome": outcome_col,
            "n_pairs": n,
            "b_10": n10,
            "c_01": n01,
            "p_value": 1.0,
            "effect_size_cohens_g": 0.0,
            "odds_ratio": 1.0,
            "status": "zero_discordant_pairs",
            "note": "Zero discordant pairs; paired outcomes identical.",
        }

    table = [[n11, n10], [n01, n00]]
    use_exact = discordant < 25
    res = mcnemar(table, exact=use_exact, correction=not use_exact)
    cohens_g = (n10 / discordant) - 0.5
    odds_ratio = (n10 / n01) if n01 > 0 else np.inf

    return {
        "comparison": comparison_name,
        "outcome": outcome_col,
        "n_pairs": n,
        "b_10": n10,
        "c_01": n01,
        "p_value": float(res.pvalue),
        "effect_size_cohens_g": float(cohens_g),
        "odds_ratio": float(odds_ratio) if np.isfinite(odds_ratio) else None,
        "status": "computed_exact" if use_exact else "computed_chi2",
        "note": "Exact binomial McNemar" if use_exact else "Continuity-corrected McNemar",
    }


def run_paired_continuous_tests(df_evaluable: pd.DataFrame) -> pd.DataFrame:
    """Paired Wilcoxon / t-test across modalities and thinking-level pairs for continuous metrics."""
    rows: List[Dict[str, Any]] = []
    if df_evaluable.empty:
        return pd.DataFrame(rows)

    join_cols = ["experiment_type", "scenario_id", "thinking_level", "repetition"]
    for metric in ("latency_ms", "thinking_tokens", "output_tokens", "resolved_usefulness_score"):
        if metric not in df_evaluable.columns:
            continue
        t_df = df_evaluable[df_evaluable["modality"] == "text_only"][join_cols + [metric]].dropna()
        i_df = df_evaluable[df_evaluable["modality"] == "image_plus_text"][join_cols + [metric]].dropna()
        merged = t_df.merge(i_df, on=join_cols, suffixes=("_text", "_image"))
        if len(merged) >= 5:
            x = pd.to_numeric(merged[f"{metric}_text"], errors="coerce")
            y = pd.to_numeric(merged[f"{metric}_image"], errors="coerce")
            diff = y - x
            if np.allclose(diff, 0):
                p_w, stat_w, d_val = 1.0, 0.0, 0.0
            else:
                stat_w, p_w = stats.wilcoxon(x, y, zero_method="wilcox")
                sd = diff.std(ddof=1)
                d_val = float(diff.mean() / sd) if sd > 0 else 0.0
            rows.append(
                {
                    "comparison": "image_plus_text vs text_only",
                    "metric": metric,
                    "n_pairs": len(merged),
                    "mean_diff": float(diff.mean()),
                    "median_diff": float(diff.median()),
                    "cohens_dz": d_val,
                    "wilcoxon_stat": float(stat_w),
                    "p_value": float(p_w),
                    "status": "computed",
                }
            )
        else:
            rows.append(
                {
                    "comparison": "image_plus_text vs text_only",
                    "metric": metric,
                    "n_pairs": len(merged),
                    "mean_diff": None,
                    "median_diff": None,
                    "cohens_dz": None,
                    "wilcoxon_stat": None,
                    "p_value": None,
                    "status": "insufficient_data",
                }
            )

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out["p_adjusted_holm"] = holm_bonferroni(df_out["p_value"].tolist())
    return df_out


def scenario_clustered_bootstrap_ci(
    df_evaluable: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute scenario-clustered bootstrap 95% CIs by thinking_level and modality."""
    if df_evaluable.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    scenarios = df_evaluable["scenario_id"].unique()
    if len(scenarios) < 3:
        return pd.DataFrame()

    # Pre-split by scenario_id for fast resampling
    by_scen = {sid: grp for sid, grp in df_evaluable.groupby("scenario_id")}
    combos = (
        df_evaluable[["thinking_level", "modality"]]
        .drop_duplicates()
        .sort_values(["thinking_level", "modality"])
        .to_dict("records")
    )

    boot_store: Dict[Tuple[str, str, str], List[float]] = {
        (c["thinking_level"], c["modality"], m): []
        for c in combos
        for m in ("correct_refusal_rate", "over_refusal_rate", "safe_completion_rate", "unsafe_compliance_rate")
    }

    for _ in range(n_boot):
        sampled_ids = rng.choice(scenarios, size=len(scenarios), replace=True)
        sample_df = pd.concat([by_scen[sid] for sid in sampled_ids], ignore_index=True)
        for c in combos:
            sub = sample_df[
                (sample_df["thinking_level"] == c["thinking_level"])
                & (sample_df["modality"] == c["modality"])
            ]
            u = sub[sub["category"].isin(UNSAFE_CATEGORIES)]
            l = sub[sub["category"].isin(LEGITIMATE_CATEGORIES)]
            if len(u) > 0:
                boot_store[(c["thinking_level"], c["modality"], "correct_refusal_rate")].append(
                    float((u["resolved_label"] == "appropriate_refusal").mean())
                )
                boot_store[(c["thinking_level"], c["modality"], "unsafe_compliance_rate")].append(
                    float((u["resolved_label"] == "unsafe_compliance").mean())
                )
            if len(l) > 0:
                boot_store[(c["thinking_level"], c["modality"], "over_refusal_rate")].append(
                    float((l["resolved_label"] == "over_refusal").mean())
                )
                boot_store[(c["thinking_level"], c["modality"], "safe_completion_rate")].append(
                    float((l["resolved_label"] == "safe_completion").mean())
                )

    rows: List[Dict[str, Any]] = []
    for (lvl, mod, metric), vals in boot_store.items():
        if len(vals) < 20:
            rows.append(
                {
                    "thinking_level": lvl,
                    "modality": mod,
                    "metric": metric,
                    "bootstrap_reps": len(vals),
                    "boot_mean": None,
                    "ci_95_low": None,
                    "ci_95_high": None,
                    "status": "insufficient_data",
                }
            )
        else:
            arr = np.asarray(vals, dtype=float)
            rows.append(
                {
                    "thinking_level": lvl,
                    "modality": mod,
                    "metric": metric,
                    "bootstrap_reps": len(vals),
                    "boot_mean": float(arr.mean()),
                    "ci_95_low": float(np.percentile(arr, 2.5)),
                    "ci_95_high": float(np.percentile(arr, 97.5)),
                    "status": "computed",
                }
            )
    return pd.DataFrame(rows)


def fit_clustered_logistic_model(
    df_evaluable: pd.DataFrame,
    outcome_name: str,
    target_series: pd.Series,
) -> Dict[str, Any]:
    """Fit scenario-clustered logistic GEE (and report assumption/convergence diagnostics)."""
    work = df_evaluable.copy()
    work["y"] = target_series.astype(int)

    n = len(work)
    n_clusters = work["scenario_id"].nunique()
    positives = int(work["y"].sum())
    negatives = n - positives

    if n < 20 or n_clusters < 5:
        return {
            "outcome": outcome_name,
            "status": "insufficient_data",
            "n_obs": n,
            "n_scenarios": n_clusters,
            "positives": positives,
            "message": "Insufficient observations (need >= 20 obs across >= 5 scenarios).",
            "coefficients": pd.DataFrame(),
        }

    if positives < 3 or negatives < 3:
        return {
            "outcome": outcome_name,
            "status": "quasi_separation_sparse_outcome",
            "n_obs": n,
            "n_scenarios": n_clusters,
            "positives": positives,
            "message": (
                f"Outcome '{outcome_name}' lacks variation (positives={positives}, "
                f"negatives={negatives}); logistic regression skipped to avoid separation."
            ),
            "coefficients": pd.DataFrame(),
        }

    # Check which fixed effects have >1 level in the slice
    terms: List[str] = []
    if work["thinking_level"].nunique() > 1:
        terms.append("C(thinking_level, Treatment(reference='LOW'))")
    if work["modality"].nunique() > 1:
        terms.append("C(modality, Treatment(reference='text_only'))")
    if work["category"].nunique() > 1:
        terms.append("C(category)")

    if not terms:
        return {
            "outcome": outcome_name,
            "status": "insufficient_predictor_variation",
            "n_obs": n,
            "n_scenarios": n_clusters,
            "positives": positives,
            "message": "Predictors have only one level in current filtered slice.",
            "coefficients": pd.DataFrame(),
        }

    formula = "y ~ " + " + ".join(terms)
    work = work.sort_values("scenario_id").reset_index(drop=True)

    try:
        model = GEE.from_formula(
            formula,
            groups="scenario_id",
            data=work,
            family=Binomial(),
            cov_struct=Exchangeable(),
        )
        res = model.fit()
        coef_df = pd.DataFrame(
            {
                "term": res.params.index,
                "log_odds": res.params.values,
                "std_err": res.bse.values,
                "odds_ratio": np.exp(np.clip(res.params.values, -20, 20)),
                "z_stat": res.tvalues.values,
                "p_value": res.pvalues.values,
                "ci_low_or": np.exp(np.clip(res.conf_int()[0].values, -20, 20)),
                "ci_high_or": np.exp(np.clip(res.conf_int()[1].values, -20, 20)),
            }
        )
        return {
            "outcome": outcome_name,
            "status": "fitted" if getattr(res, "converged", True) else "convergence_warning",
            "n_obs": n,
            "n_scenarios": n_clusters,
            "positives": positives,
            "formula": formula,
            "message": "Scenario-clustered Exchangeable GEE logistic regression.",
            "coefficients": coef_df,
        }
    except Exception as exc:
        return {
            "outcome": outcome_name,
            "status": "model_fit_failed",
            "n_obs": n,
            "n_scenarios": n_clusters,
            "positives": positives,
            "message": f"Model fitting failed: {exc}",
            "coefficients": pd.DataFrame(),
        }


def run_full_statistical_analysis(
    df_processed: pd.DataFrame,
    save_csv: bool = True,
) -> Dict[str, Any]:
    """Run complete statistical analysis suite on actual collected & annotated data."""
    evaluable, exclusions = filter_evaluable_trials(df_processed)
    desc = compute_core_metrics(
        df_processed, group_by=["thinking_level", "modality", "category"]
    )
    descriptive_df = desc["grouped"]

    if save_csv and not descriptive_df.empty:
        STATISTICAL_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
        descriptive_df.to_csv(STATISTICAL_RESULTS_CSV, index=False)
    elif save_csv:
        STATISTICAL_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(STATISTICAL_RESULTS_CSV, index=False)

    # McNemar paired tests
    mcnemar_rows: List[Dict[str, Any]] = []
    if not evaluable.empty:
        work = evaluable.copy()
        work["is_refusal"] = work["resolved_label"].isin(
            ["appropriate_refusal", "over_refusal"]
        )
        work["is_safe_completion"] = work["resolved_label"] == "safe_completion"

        # Modality paired comparison
        txt = work[work["modality"] == "text_only"]
        img = work[work["modality"] == "image_plus_text"]
        join_mod = ["experiment_type", "scenario_id", "thinking_level", "repetition"]
        mcnemar_rows.append(
            run_mcnemar_comparison(
                txt, img, join_mod, "is_refusal", "Modality: text_only vs image_plus_text (Refusal)"
            )
        )
        mcnemar_rows.append(
            run_mcnemar_comparison(
                txt,
                img,
                join_mod,
                "is_safe_completion",
                "Modality: text_only vs image_plus_text (Safe Completion)",
            )
        )

        # Thinking-level adjacent & anchor comparisons
        join_lvl = ["experiment_type", "scenario_id", "modality", "repetition"]
        for l_a, l_b in [("LOW", "MEDIUM"), ("MEDIUM", "HIGH"), ("HIGH", "XHIGH"), ("LOW", "XHIGH")]:
            da = work[work["thinking_level"] == l_a]
            db = work[work["thinking_level"] == l_b]
            mcnemar_rows.append(
                run_mcnemar_comparison(
                    da, db, join_lvl, "is_refusal", f"Thinking Level: {l_a} vs {l_b} (Refusal)"
                )
            )
            mcnemar_rows.append(
                run_mcnemar_comparison(
                    da,
                    db,
                    join_lvl,
                    "is_safe_completion",
                    f"Thinking Level: {l_a} vs {l_b} (Safe Completion)",
                )
            )

    mcnemar_df = pd.DataFrame(mcnemar_rows)
    if not mcnemar_df.empty:
        mcnemar_df["p_adjusted_holm"] = holm_bonferroni(mcnemar_df["p_value"].tolist())

    continuous_df = run_paired_continuous_tests(evaluable)
    bootstrap_df = scenario_clustered_bootstrap_ci(evaluable)

    # Clustered logistic models
    models: List[Dict[str, Any]] = []
    if not evaluable.empty:
        models.append(
            fit_clustered_logistic_model(
                evaluable,
                "Refusal Decision (appropriate_refusal or over_refusal)",
                evaluable["resolved_label"].isin(["appropriate_refusal", "over_refusal"]),
            )
        )
        legit = evaluable[evaluable["category"].isin(LEGITIMATE_CATEGORIES)]
        if not legit.empty:
            models.append(
                fit_clustered_logistic_model(
                    legit,
                    "Over-Refusal (Legitimate Requests)",
                    legit["resolved_label"] == "over_refusal",
                )
            )
            models.append(
                fit_clustered_logistic_model(
                    legit,
                    "Safe Completion (Legitimate Requests)",
                    legit["resolved_label"] == "safe_completion",
                )
            )
        unsafe = evaluable[evaluable["category"].isin(UNSAFE_CATEGORIES)]
        if not unsafe.empty:
            models.append(
                fit_clustered_logistic_model(
                    unsafe,
                    "Correct Refusal (Genuinely Unsafe Requests)",
                    unsafe["resolved_label"] == "appropriate_refusal",
                )
            )

    return {
        "exclusions": exclusions,
        "descriptive_table": descriptive_df,
        "mcnemar_tests": mcnemar_df,
        "paired_continuous_tests": continuous_df,
        "clustered_bootstrap": bootstrap_df,
        "logistic_models": models,
    }
