# RET-C2-337 — REAL TimesFM 2.5 local backend integration test.
#
# Runs the actual TimesFM_2p5_200M_torch model end-to-end through TimesFMService
# and the full agent. SKIPPED in CI (the `local` extra is not installed there —
# run-tests installs only [dev], so the deterministic `baseline` backend keeps
# CI green). Run locally with: pip install ".[local]" && pytest -m real_model.
#
# This test exists so the real-model integration is verified by *running it*,
# not just by a stub. Verified on CPU (forecast + forecast_with_covariates,
# xreg via CPU jax). The model load is slow (~60s cold), so it is opt-in.

import json

import pytest

# Skip the whole module unless the heavy optional deps are importable.
pytest.importorskip("timesfm", reason="local extra not installed (CI uses baseline)")
pytest.importorskip("jax", reason="xreg covariate path needs jax (CPU ok)")

from framework.schemas.invocation_context import InvocationContext  # noqa: E402
from framework.schemas.trust_level import TrustLevel  # noqa: E402

from src.graph.graph import RETDemandAnomalyAgent  # noqa: E402
from src.services.timesfm_service import TimesFMService, _local_backend  # noqa: E402


pytestmark = pytest.mark.real_model


def _ctx():
    return InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)


def test_local_backend_runs_on_cpu():
    """The real TimesFM service produces point + CI bands for each SKU (CPU)."""
    svc = TimesFMService(backend=_local_backend)
    series = [float(x) for x in ([10, 11, 9, 10, 12, 10, 11] * 8)]
    out = svc.forecast({"SKU1": series}, {"promo": [0.0] * len(series)}, horizon=7, inference_mode="local")
    pt = out["forecast"]["SKU1"]
    band = out["confidence_interval"]["SKU1"]
    assert len(pt) == 7
    assert len(band["lower"]) == 7 and len(band["upper"]) == 7
    for lo, p, hi in zip(band["lower"], pt, band["upper"]):
        assert lo <= hi
    # stable ~10-11 series -> forecast should land in a sane range
    assert 5.0 < (sum(pt) / len(pt)) < 16.0


def test_full_agent_with_real_local_model():
    """Full invoke with inference_mode='local' completes through the real model.

    The covariate_forecast node's TimesFMService resolves the `local` backend
    from inference_mode automatically — no injection needed.
    """
    agent = RETDemandAnomalyAgent()
    agent.compile()
    req = {
        "series": {"SKU2": [10, 11, 9, 10, 12, 10, 11, 10, 11, 9, 10, 12, 10, 40] * 2},
        "covariates": {"promo": [0.0] * 28},
        "target_skus": ["SKU2"],
        "inference_mode": "local",
        "horizon": 7,
    }
    out = agent.invoke(json.dumps(req), ctx=_ctx())["output"]
    # The deterministic detection layer still runs on the real forecast.
    assert "anomaly_detected" in out
    assert isinstance(out["report_markdown"], str) and out["report_markdown"]
