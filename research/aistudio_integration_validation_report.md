# Google AI Studio Integration & Experiment Workflow Validation Report

- **Generated (UTC):** 2026-11-14T19:51:07.618212+00:00
- **Software Version:** `1.0.0`
- **Google GenAI SDK:** `2.33.0`
- **Target Model:** `gemini-4-hydrogen`

## 1. Separation of Scenario Datasets vs. Experimental Observations
- **Scenario Datasets (Versioned Inputs):** `dataset/versions/<version>/scenarios.json` & `.jsonl` (read-only inside Streamlit; curated in Google AI Studio and imported via JSON, JSONL, or CSV).
- **Pilot Observation Archive (Authoritative Append-Only JSONL):** `/app/applet/results/pilot/raw_responses.jsonl` (current records: `0`).
- **Main Observation Archive (Authoritative Append-Only JSONL):** `/app/applet/results/main/raw_responses.jsonl` (current records: `0`).

## 2. Google AI Studio API & Logging Verification Status
- **`GEMINI_API_KEY` Configured:** `False`
- **Validated Thinking Levels (Live Endpoint):** `None yet`
- **Google AI Studio Project Logging Status:** **UNVERIFIED (GEMINI_API_KEY not configured in runtime)**
- **Undocumented UI Dataset Retrieval APIs Used:** `False` (the official `google-genai` SDK does not expose retrieval of arbitrary UI-named AI Studio Datasets; scenario datasets are imported explicitly as versioned JSON/JSONL/CSV files).

## 3. Immutable Scenario Version Integrity
- `v1.0.0-pilot` (`pilot`): approval=`pending_review`, sha256_intact=`True`, sha256=`5a8cf0a4e89c0c5da510747fad3d517f5b1cd2e6277689f9183ea3c778a6483d`
- `v1.0.0-main` (`main`): approval=`pending_review`, sha256_intact=`True`, sha256=`f962d716e3142553e0f3bb2182dd0c3766a8fd1121ae6d3ae152d6e94fa727fd`

## 4. Next Verification Steps
1. Configure `GEMINI_API_KEY` and enable **Project Logging** in Google AI Studio.
2. Run `python3 validate_thinking_levels.py --include-multimodal` to validate `LOW`, `MEDIUM`, `HIGH`, and `XHIGH` against `models/gemini-4-hydrogen`.
3. Confirm the resulting probe requests appear in the Google AI Studio **Logs** UI.
4. Grant explicit Researcher Approval to `v1.0.0-pilot` before running the 96-call Pilot, and wait for pilot review before executing the 1,440-call Main experiment.
