# Gemini 4 Hydrogen API Readiness Report

**Generated:** 2026-11-19 06:50:18 UTC
**Overall API Status:** `INCOMPLETE / BLOCKED (unvalidated thinking levels: LOW, MEDIUM, HIGH, XHIGH)`
**Target Model Identifier:** `gemini-4-hydrogen` (`models/gemini-4-hydrogen`)
**Google GenAI SDK Version:** `3.6.1`
**`GEMINI_API_KEY` Configured:** `True` (credential values never stored in reports or logs)

## 1. Model Endpoint Accessibility (`models.get`)
- **Accessible:** `True`
- **Endpoint Name:** `models/gemini-4-hydrogen`
- **Display Name:** `Gemini 4 Hydrogen`
- **Input / Output Token Limits:** `1048576` / `65536`
- **Error:** `None`

## 2. Thinking-Level Endpoint Support Matrix

| Thinking Level | Validated | Endpoint Supported | Modality Tested | Latency (ms) | Input / Output / Thinking Tokens | Last Tested (UTC) | Error |
|---|---|---|---|---|---|---|---|
| `LOW` | `False` | `UNTESTED` | `text_only` | 29517.8 | `Unavailable` | `2026-11-19T06:50:18.390698+00:00` | 504 DEADLINE_EXCEEDED. {'error': {'code': 504, 'message': 'Deadline expired before operation could complete.', 'status': 'DEADLINE_EXCEEDED'}} |
| `MEDIUM` | `False` | `UNTESTED` | `text_only` | 59427.6 | `Unavailable` | `2026-11-19T06:47:34.629240+00:00` | 504 DEADLINE_EXCEEDED. {'error': {'code': 504, 'message': 'Deadline expired before operation could complete.', 'status': 'DEADLINE_EXCEEDED'}} |
| `HIGH` | `False` | `UNTESTED` | `text_only` | 59807.7 | `Unavailable` | `2026-11-19T06:49:36.516371+00:00` | 504 DEADLINE_EXCEEDED. {'error': {'code': 504, 'message': 'Deadline expired before operation could complete.', 'status': 'DEADLINE_EXCEEDED'}} |
| `XHIGH` | `False` | `UNTESTED` | `None` | N/A | `Unavailable` | `Never` | None |

## 3. Multimodal (`image_plus_text`) Probe Status
- **Multimodal Probe Validated:** `False`
- **Thinking Level Used:** `None`
- **Last Tested (UTC):** `Never`
- **Error:** `None`

## 4. Verification Command
Run `.venv/bin/python validate_thinking_levels.py --include-multimodal` once `GEMINI_API_KEY` is set.
