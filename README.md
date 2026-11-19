# Gemini 4 Hydrogen Cross-Modal Safety Research Workbench

**Study Title:** *Thinking-Level Effects on Cross-Modal Safety Decisions in Gemini 4 Hydrogen: A Controlled Study of Refusal Appropriateness, Safe Completion, and Inference Cost*

## Quick Start (Python + Canonical 10-Page Streamlit Workbench)

1. Bootstrap / install Python dependencies into `.venv` (or run `bash start_dev.sh` which auto-bootstraps `.venv`):
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```
2. Configure your Gemini API key in `.env` (copy from `.env.example`; never commit secrets):
   ```bash
   GEMINI_API_KEY="your_actual_key_here"
   ```
3. Run the standalone 4-thinking-level endpoint validation script:
   ```bash
   .venv/bin/python validate_thinking_levels.py --include-multimodal
   ```
4. Run automated unit & integration tests:
   ```bash
   .venv/bin/pytest -v
   ```
5. Launch the canonical 10-page Streamlit research workbench (port `3000`):
   ```bash
   bash start_dev.sh
   ```
   Or via Python launcher:
   ```bash
   .venv/bin/python main.py --port 3000
   ```

## Workflow
1. **API Configuration (Page 2):** Verify `GEMINI_API_KEY`, probe `LOW`, `MEDIUM`, `HIGH`, and `XHIGH` against `gemini-4-hydrogen`, and generate `research/api_readiness_report.md`.
2. **Dataset Status (Page 3):** Inspect the 60 matched scenario families (non-legal), record per-scenario human reviews (`reviewer_id` + timestamp in `review_log.json` without mutating SHA-256), and approve `v1.0.0-pilot` (12) and `v1.0.0-main` (60).
3. **Pilot Experiment (Page 4):** Run the fixed 96-call pilot (`12 × 2 × 4 × 1`) and generate `research/pilot_report.md` and `results/pilot/pilot_experiment_manifest.json`.
4. **Main Experiment (Page 5):** After pilot completion/review, explicitly approve and run the fixed 1,440-call main experiment (`60 × 2 × 4 × 3`) with append-only JSONL storage and `results/main/main_experiment_manifest.json`.
5. **Response Annotation (Page 6):** Perform independent dual human coding (with pseudonymous coder IDs and timestamps), check Cohen's kappa, and adjudicate disagreements.
6. **Results, Statistics, Robustness & Export (Pages 7–10):** Inspect Charts A–G, McNemar/bootstrap/GEE/mixed-effects/power analyses, robustness checks, and export CSV/JSON/JSONL/Markdown/PNG artifacts.
