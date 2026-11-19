"""Standalone validation script independently testing LOW, MEDIUM, HIGH, and XHIGH against Gemini 4 Hydrogen."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import DEFAULT_MODEL_ID, THINKING_LEVELS, get_api_key
from dataset.scenario_manager import resolve_image_path
from experiment.hydrogen_client import (
    get_sdk_version,
    test_single_thinking_level,
    verify_model_endpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Independently verify Gemini 4 Hydrogen API access and all four thinking levels "
            "(LOW, MEDIUM, HIGH, XHIGH) without fabricating responses."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help="Model identifier (default: gemini-4-hydrogen)",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=512,
        help="Output token limit for validation probes (default: 512)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds (default: 60.0)",
    )
    parser.add_argument(
        "--include-multimodal",
        action="store_true",
        help="Also run an image-plus-text probe using dataset/images/scn-001.png",
    )
    args = parser.parse_args()

    print(f"google-genai SDK version: {get_sdk_version()}")
    api_key = get_api_key()
    if not api_key:
        print(
            "ERROR: GEMINI_API_KEY is not set in environment variables or .env. "
            "Set GEMINI_API_KEY and re-run.",
            file=sys.stderr,
        )
        return 2

    print(f"Checking model endpoint: {args.model}")
    ep = verify_model_endpoint(model_id=args.model, timeout_seconds=args.timeout)
    if not ep["accessible"]:
        print(f"Endpoint check failed: {ep['error']}", file=sys.stderr)
        return 1
    print(
        f"Endpoint accessible: {ep['endpoint_name']} | "
        f"Input limit: {ep['input_token_limit']} | Output limit: {ep['output_token_limit']}"
    )

    all_ok = True
    for level in THINKING_LEVELS:
        print(f"\n--- Testing thinking_level={level} (text_only) ---")
        res = test_single_thinking_level(
            thinking_level=level,
            model_id=args.model,
            max_output_tokens=args.max_output_tokens,
            timeout_seconds=args.timeout,
        )
        print(
            f"Status: {res.status} | Actual Model: {res.actual_model_version} | "
            f"Latency: {res.latency_ms} ms"
        )
        print(
            f"Tokens -> input={res.input_tokens}, output={res.output_tokens}, "
            f"thinking={res.thinking_tokens}, total={res.total_tokens}"
        )
        if res.status == "success":
            print(f"Response preview: {res.response_text[:180]}")
        else:
            all_ok = False
            print(f"Error ({res.error_code}): {res.error_message}", file=sys.stderr)

    if args.include_multimodal:
        img_path = resolve_image_path("scn-001.png")
        print(f"\n--- Testing multimodal image_plus_text probe ({img_path}) ---")
        mm_res = test_single_thinking_level(
            thinking_level="HIGH",
            model_id=args.model,
            max_output_tokens=args.max_output_tokens,
            timeout_seconds=args.timeout,
            image_path=Path(img_path) if img_path else None,
        )
        print(
            f"Multimodal status: {mm_res.status} | Latency: {mm_res.latency_ms} ms | "
            f"Tokens -> in={mm_res.input_tokens}, out={mm_res.output_tokens}, think={mm_res.thinking_tokens}"
        )
        if mm_res.status != "success":
            all_ok = False
            print(f"Multimodal error: {mm_res.error_message}", file=sys.stderr)

    print("\nValidation summary:", "ALL 4 LEVELS VALIDATED" if all_ok else "VALIDATION INCOMPLETE / FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
