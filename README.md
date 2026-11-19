# Gemini 4 Hydrogen Cross-Modal Safety Research Workbench

**Study Title:** *Thinking-Level Effects on Cross-Modal Safety Decisions in Gemini 4 Hydrogen: A Controlled Study of Refusal Appropriateness, Safe Completion, and Inference Cost*

## Quick Start (Local Python + Streamlit)

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure your Gemini API key in `.env` (copy from `.env.example`):
   ```bash
   GEMINI_API_KEY="your_actual_key_here"
   ```
3. Run automated unit tests:
   ```bash
   pytest -v
   ```
4. Launch the Streamlit research workbench:
   ```bash
   streamlit run app/streamlit_app.py
   ```
   Or via Python launcher:
   ```bash
   python main.py
   ```

## Workflow
1. **API Configuration (Page 2):** Verify `GEMINI_API_KEY` and test `LOW`, `MEDIUM`, `HIGH`, and `XHIGH` against `gemini-4-hydrogen`.
2. **Dataset Management (Page 3):** Review the 60 matched scenario families (non-legal) and approve scenarios for pilot and main phases.
3. **Pilot Experiment (Page 4):** Run the 96-call pilot (`12 × 2 × 4 × 1`) and generate `research/pilot_report.md`.
4. **Main Experiment (Page 5):** Explicitly approve and run the 1,440-call main experiment (`60 × 2 × 4 × 3`) with append-only JSONL storage and resumability.
5. **Response Annotation (Page 6):** Perform independent dual human coding, check Cohen's kappa, and adjudicate disagreements.
6. **Results, Statistics, Robustness & Export (Pages 7–10):** Inspect Charts A–G, McNemar/bootstrap/GEE analyses, robustness checks, and export CSV/JSON/PNG artifacts.
