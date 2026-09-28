"""Shared fixtures for RET-C2-337 tests and the live-LLM smoke check."""

from __future__ import annotations

import pytest


@pytest.fixture
def agent_config() -> dict:
    """Extra construction config for the live-LLM smoke check.

    Nothing is needed here: the harness puts the live client into
    `config["llm"]`, which `RETDemandAnomalyAgent.register_nodes()` forwards to
    the inner workflow and on to `ReportGenerateNode` (the production seam).
    Bridging the LLM in from tests would verify a test-authored prompt instead
    of the template's own, so it is deliberately NOT done here.
    """
    return {}


@pytest.fixture
def case_payload() -> dict:
    """`tests/integration/test_end_to_end.py::test_happy_path_detects_spike` と同形の入力。"""
    return {
        "series": {
            "SKU1": [10, 11, 9, 10, 12, 10, 11] * 2,
            "SKU2": [10, 11, 9, 10, 12, 10, 11, 10, 11, 9, 10, 12, 10, 40],
        },
        "covariates": {"promo": [0] * 14},
        "target_skus": ["SKU1", "SKU2"],
        "period": "2026-06",
        "inference_mode": "baseline",
        "horizon": 7,
    }


@pytest.fixture
def live_input_context() -> dict:
    return {"mode": "invoke"}
