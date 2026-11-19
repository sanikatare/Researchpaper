"""API configuration and Gemini 4 Hydrogen client unit tests (mock clients only; never live findings)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors
from google.genai import types

from config import DEFAULT_MODEL_ID, THINKING_LEVELS
from dataset.scenario_manager import resolve_image_path
from experiment import hydrogen_client
from experiment.hydrogen_client import (
    MissingApiKeyError,
    build_client,
    call_gemini_hydrogen,
    normalize_model_identifier,
    resolve_thinking_config,
    verify_model_endpoint,
)


class _FakeModels:
    def __init__(self, responses=None, fail_levels=None) -> None:
        self.calls = []
        self._responses = list(responses or [])
        self._fail_levels = dict(fail_levels or {})

    def get(self, model: str):
        return SimpleNamespace(
            name=f"models/{model}",
            display_name="Gemini 4 Hydrogen",
            supported_actions=["generateContent"],
            input_token_limit=1048576,
            output_token_limit=65536,
        )

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        lvl = str(config.thinking_config.thinking_level).split(".")[-1]
        if lvl in self._fail_levels:
            raise self._fail_levels[lvl]
        if self._responses:
            nxt = self._responses.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt

        part = SimpleNamespace(text=f"Verified output for {lvl}.", thought=False)
        cand = SimpleNamespace(finish_reason="STOP", content=SimpleNamespace(parts=[part]))
        usage = SimpleNamespace(
            prompt_token_count=25,
            candidates_token_count=12,
            thoughts_token_count=64,
            total_token_count=101,
        )
        return SimpleNamespace(
            text=f"Verified output for {lvl}.",
            model_version="gemini-4-hydrogen-001",
            response_id="resp-123",
            usage_metadata=usage,
            prompt_feedback=None,
            candidates=[cand],
        )


class _FakeClient:
    def __init__(self, responses=None, fail_levels=None) -> None:
        self.models = _FakeModels(responses=responses, fail_levels=fail_levels)


def test_model_identifier_normalization_and_restriction() -> None:
    assert normalize_model_identifier("gemini-4-hydrogen") == DEFAULT_MODEL_ID
    assert normalize_model_identifier("models/gemini-4-hydrogen") == DEFAULT_MODEL_ID
    with pytest.raises(ValueError):
        normalize_model_identifier("gemini-1.5-pro")


def test_missing_api_key_raises_cleanly(monkeypatch) -> None:
    monkeypatch.setattr(hydrogen_client, "get_api_key", lambda: None)
    with pytest.raises(MissingApiKeyError):
        build_client(api_key=None)


def test_thinking_config_supports_four_levels_without_deprecated_params() -> None:
    for level in THINKING_LEVELS:
        cfg = resolve_thinking_config(level)
        assert str(cfg.thinking_level).endswith(level)
        assert cfg.thinking_budget is None


def test_text_and_multimodal_requests_and_token_metadata() -> None:
    fake = _FakeClient()
    res_txt = call_gemini_hydrogen(
        prompt="Explain deionized water rinsing.",
        thinking_level="HIGH",
        model_id="models/gemini-4-hydrogen",
        max_output_tokens=1024,
        client=fake,
    )
    assert res_txt.status == "success"
    assert res_txt.model_id == DEFAULT_MODEL_ID
    assert res_txt.actual_model_version == "gemini-4-hydrogen-001"
    assert res_txt.latency_ms >= 0.0
    assert res_txt.input_tokens == 25
    assert res_txt.output_tokens == 12
    assert res_txt.thinking_tokens == 64
    assert res_txt.total_tokens == 101

    sent_cfg = fake.models.calls[0]["config"]
    assert sent_cfg.max_output_tokens == 1024
    assert sent_cfg.temperature is None
    assert sent_cfg.top_p is None

    img_path = resolve_image_path("scn-001.png")
    res_img = call_gemini_hydrogen(
        prompt="Read the thermometer image.",
        thinking_level="XHIGH",
        image_path=img_path,
        client=fake,
    )
    assert res_img.status == "success"
    assert res_img.modality == "image_plus_text"
    contents = fake.models.calls[1]["contents"]
    assert isinstance(contents, list)
    assert isinstance(contents[0], types.Part)


def test_unavailable_usage_metadata_never_fabricated() -> None:
    part = SimpleNamespace(text="Response without usage metadata.", thought=False)
    cand = SimpleNamespace(finish_reason="STOP", content=SimpleNamespace(parts=[part]))
    resp = SimpleNamespace(
        text="Response without usage metadata.",
        model_version="gemini-4-hydrogen",
        response_id="resp-no-usage",
        usage_metadata=None,
        prompt_feedback=None,
        candidates=[cand],
    )
    fake = _FakeClient(responses=[resp])
    res = call_gemini_hydrogen(prompt="Hello", thinking_level="LOW", client=fake)
    assert res.status == "success"
    assert res.input_tokens is None
    assert res.output_tokens is None
    assert res.thinking_tokens is None
    assert res.total_tokens is None


def test_controlled_retries_and_error_logging(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "api_errors.log"
    monkeypatch.setattr(hydrogen_client, "API_ERROR_LOG_PATH", log_path)
    # Reset logger handlers so it binds to temp log_path
    logger = logging.getLogger("gemini4hydrogen.api")
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    ok_part = SimpleNamespace(text="Recovered after transient rate limit.", thought=False)
    ok_resp = SimpleNamespace(
        text="Recovered after transient rate limit.",
        model_version="gemini-4-hydrogen",
        response_id="resp-ok",
        usage_metadata=None,
        prompt_feedback=None,
        candidates=[SimpleNamespace(finish_reason="STOP", content=SimpleNamespace(parts=[ok_part]))],
    )
    fake = _FakeClient(responses=[TimeoutError("429 RESOURCE_EXHAUSTED retry in 0.01s"), ok_resp])
    res = call_gemini_hydrogen(
        prompt="Test retry",
        thinking_level="MEDIUM",
        max_retries=2,
        retry_base_delay=0.01,
        client=fake,
    )
    assert res.status == "success"
    assert res.retry_count == 1
    assert log_path.exists()
    assert "RESOURCE_EXHAUSTED" in log_path.read_text(encoding="utf-8")


def test_endpoint_unsupported_level_is_recorded_never_simulated(tmp_path, monkeypatch) -> None:
    val_path = tmp_path / "api_validation_state.json"
    monkeypatch.setattr(hydrogen_client, "API_VALIDATION_STATE_PATH", val_path)
    monkeypatch.setattr(hydrogen_client, "API_ERROR_LOG_PATH", tmp_path / "api_errors.log")

    err_400 = genai_errors.ClientError(
        code=400,
        response_json={"error": {"message": "Thinking level XHIGH is not supported on this endpoint."}},
        response=None,
    )
    fake = _FakeClient(fail_levels={"XHIGH": err_400})
    results = hydrogen_client.test_all_thinking_levels(max_retries=0, client=fake)
    by_lvl = {r.thinking_level: r for r in results}
    assert by_lvl["LOW"].status == "success"
    assert by_lvl["MEDIUM"].status == "success"
    assert by_lvl["HIGH"].status == "success"
    assert by_lvl["XHIGH"].status == "error"

    state = hydrogen_client.load_validation_state()
    assert state["levels"]["LOW"]["validated"] is True
    assert state["levels"]["XHIGH"]["validated"] is False
    assert state["levels"]["XHIGH"]["supported_by_endpoint"] is False
    assert hydrogen_client.is_api_fully_validated() is False


def test_verify_model_endpoint_metadata() -> None:
    fake = _FakeClient()
    meta = verify_model_endpoint("models/gemini-4-hydrogen", client=fake)
    assert meta["accessible"] is True
    assert meta["canonical_model"] == DEFAULT_MODEL_ID
    assert meta["endpoint_name"] == "models/gemini-4-hydrogen"
    assert meta["input_token_limit"] == 1048576
    assert meta["output_token_limit"] == 65536


def test_timeout_validation_and_http_options(monkeypatch) -> None:
    with pytest.raises(ValueError):
        build_client(api_key="test-key-1234", timeout_seconds=0)

    captured = {}

    class _DummyGenAIClient:
        def __init__(self, api_key, http_options):
            captured["api_key"] = api_key
            captured["http_options"] = http_options

    monkeypatch.setattr(hydrogen_client.genai, "Client", _DummyGenAIClient)
    build_client(api_key="test-key-1234", timeout_seconds=45.5)
    assert captured["api_key"] == "test-key-1234"
    assert captured["http_options"].timeout == 45500


def test_rate_limit_pacing_and_output_token_bounds(monkeypatch) -> None:
    monkeypatch.setattr(hydrogen_client, "_LAST_REQUEST_MONOTONIC", 0.0)
    fake = _FakeClient()

    bad_tok = call_gemini_hydrogen(
        prompt="Test invalid token bound",
        thinking_level="LOW",
        max_output_tokens=70000,
        client=fake,
    )
    assert bad_tok.status == "error"
    assert bad_tok.error_code == "INVALID_MAX_OUTPUT_TOKENS"

    import time

    t0 = time.monotonic()
    r1 = call_gemini_hydrogen(
        prompt="Call 1",
        thinking_level="LOW",
        min_request_interval=0.06,
        client=fake,
    )
    r2 = call_gemini_hydrogen(
        prompt="Call 2",
        thinking_level="LOW",
        min_request_interval=0.06,
        client=fake,
    )
    elapsed = time.monotonic() - t0
    assert r1.status == "success"
    assert r2.status == "success"
    assert elapsed >= 0.05


def test_retry_exhaustion_non_retryable_and_blocked(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "api_errors.log"
    monkeypatch.setattr(hydrogen_client, "API_ERROR_LOG_PATH", log_path)
    logger = logging.getLogger("gemini4hydrogen.api")
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    # Transient errors exhaust retries cleanly
    fake_exhaust = _FakeClient(
        responses=[
            TimeoutError("503 UNAVAILABLE"),
            TimeoutError("503 UNAVAILABLE"),
            TimeoutError("503 UNAVAILABLE"),
        ]
    )
    res_exhaust = call_gemini_hydrogen(
        prompt="Transient failure",
        thinking_level="HIGH",
        max_retries=2,
        retry_base_delay=0.005,
        client=fake_exhaust,
    )
    assert res_exhaust.status == "error"
    assert res_exhaust.retry_count == 2
    assert len(fake_exhaust.models.calls) == 3

    # Non-retryable 400 ClientError does not retry
    err_400 = genai_errors.ClientError(
        code=400,
        response_json={"error": {"message": "Bad request payload"}},
        response=None,
    )
    fake_400 = _FakeClient(responses=[err_400])
    res_400 = call_gemini_hydrogen(
        prompt="Bad request",
        thinking_level="LOW",
        max_retries=3,
        retry_base_delay=0.005,
        client=fake_400,
    )
    assert res_400.status == "error"
    assert res_400.retry_count == 0
    assert len(fake_400.models.calls) == 1

    # Blocked prompt_feedback returns blocked without fabricated text
    blocked_resp = SimpleNamespace(
        text="",
        model_version="gemini-4-hydrogen",
        response_id="resp-blocked",
        usage_metadata=None,
        prompt_feedback=SimpleNamespace(block_reason="SAFETY"),
        candidates=[],
    )
    fake_blocked = _FakeClient(responses=[blocked_resp])
    res_blocked = call_gemini_hydrogen(
        prompt="Blocked probe",
        thinking_level="MEDIUM",
        client=fake_blocked,
    )
    assert res_blocked.status == "blocked"
    assert res_blocked.block_reason == "SAFETY"
    assert res_blocked.response_text == ""


def test_api_readiness_report_and_multimodal_probe(tmp_path, monkeypatch) -> None:
    val_path = tmp_path / "api_validation_state.json"
    monkeypatch.setattr(hydrogen_client, "API_VALIDATION_STATE_PATH", val_path)
    monkeypatch.setattr(hydrogen_client, "get_api_key", lambda: "AIzaFakeTestKey123456")

    fake = _FakeClient()
    hydrogen_client.test_all_thinking_levels(max_retries=0, client=fake)
    img_path = resolve_image_path("scn-001.png")
    hydrogen_client.test_single_thinking_level("HIGH", image_path=img_path, max_retries=0, client=fake)

    state = hydrogen_client.load_validation_state()
    assert state["levels"]["LOW"]["modality_tested"] == "text_only"
    assert state["multimodal_probe"]["validated"] is True
    assert state["multimodal_probe"]["image_asset"] == "scn-001.png"

    rep_path = tmp_path / "api_readiness_report.md"
    ep = verify_model_endpoint(client=fake)
    rep_md = hydrogen_client.generate_api_readiness_report(save_path=rep_path, endpoint_check=ep)
    assert rep_path.exists()
    assert "READY (All 4 thinking levels live-validated)" in rep_md
    assert "AIzaFakeTestKey123456" not in rep_md

    # When API key is missing, report must honestly say BLOCKED
    monkeypatch.setattr(hydrogen_client, "get_api_key", lambda: None)
    rep_blocked = hydrogen_client.generate_api_readiness_report(save_path=rep_path)
    assert "BLOCKED (GEMINI_API_KEY not configured" in rep_blocked

