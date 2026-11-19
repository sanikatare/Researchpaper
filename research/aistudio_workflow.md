# Google AI Studio Dataset Authoring, Logging & Curation Workflow

## 1. Scenario Authoring & Revision (Google AI Studio)
All scenario definitions (`scenario_id`, `category`, `scenario_title`, `scenario_description`, `underlying_user_intent`, `text_only_prompt`, `image_text_prompt`, `image_asset_reference`, `image_provenance`, `expected_safe_behavior`, `expected_response_classification`, `classification_rationale`, `dataset_version`, `validation_status`) are authored and revised exclusively inside **Google AI Studio** using structured JSON/JSONL output and `dataset/aistudio_prompt_template.md`.

The Streamlit **Dataset Status** page is strictly read-only for scenario content: it imports versioned JSON/JSONL artifacts, validates all constraints, tracks version history, and manages separate Pilot vs. Main active versions and researcher approval.

## 2. Google AI Studio Logs & Datasets Capabilities (Verified Boundary)
- **Supported in Google AI Studio:**
  - Enabling **Project Logging** captures `GenerateContent` API requests and responses (both successful and failed) automatically for the associated API project.
  - From the **Logs** dashboard, researchers can filter requests/responses and curate them into named **Datasets** for regression testing or offline evaluation, and export those datasets as **CSV**, **JSONL**, or **Google Sheets**.
- **Important Technical Limitation (Respected by This Engine):**
  - The Gemini Python SDK does **not** retrieve arbitrary UI-created Google AI Studio Datasets by their UI name, nor does the Logs dashboard replace controlled factorial scenario authoring.
  - Therefore:
    1. **Scenario inputs** are authored in Google AI Studio, exported as JSON/JSONL, and imported into `dataset/versions/` as immutable versioned files.
    2. **Experimental observations** are always persisted locally by the Python engine in append-only JSONL (`results/pilot/raw_responses.jsonl` and `results/main/raw_responses.jsonl`), with every trial tagged with `dataset_version` and `dataset_name`.
    3. When project logging is enabled in Google AI Studio, live API calls also appear in AI Studio Logs for secondary audit and dataset curation/export (CSV / JSONL / Google Sheets). If logging is unavailable on a given account tier, local append-only JSONL remains the authoritative immutable record and the UI never pretends AI Studio logging succeeded.
