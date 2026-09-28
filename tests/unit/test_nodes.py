# RET-C2-337 — Unit Tests: detection nodes (deterministic) + node contract

import inspect

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.services.llm.base_llm import BaseLLM

from src.nodes.pre_process_node import PreProcessNode
from src.nodes.data_normalize_node import DataNormalizeNode
from src.nodes.covariate_forecast_node import CovariateForecastNode
from src.nodes.stat_anomaly_detect_node import StatAnomalyDetectNode
from src.nodes.event_attribution_node import EventAttributionNode
from src.nodes.report_generate_node import ReportGenerateNode


def _base_state(**extra):
    state = {"node_history": [], "error_log": []}
    state.update(extra)
    return state


class TestNodeContract:
    """node contract: each node implements execute(self, state); no _invoke_impl."""

    NODES = [
        PreProcessNode,
        DataNormalizeNode,
        CovariateForecastNode,
        StatAnomalyDetectNode,
        EventAttributionNode,
        ReportGenerateNode,
    ]

    def test_execute_signature(self):
        for node_cls in self.NODES:
            assert hasattr(node_cls, "execute"), f"{node_cls.__name__} must implement execute()"
            params = list(inspect.signature(node_cls.execute).parameters)
            assert params[1] == "state", f"{node_cls.__name__}.execute second param must be 'state'"
            assert "_invoke_impl" not in node_cls.__dict__, f"{node_cls.__name__} must not define _invoke_impl"


class TestDataNormalize:
    def test_forward_fill_and_parse(self):
        import json

        req = {"series": {"A": [1, None, 3]}, "covariates": {"promo": [0, 1]}, "target_skus": ["A"]}
        out = DataNormalizeNode().execute(_base_state(user_input=json.dumps(req)))
        assert out["status"] == AgentStatus.SUCCESS
        assert out["normalized_series"]["A"] == [1.0, 1.0, 3.0]  # None forward-filled
        assert out["covariate_matrix"]["promo"] == [0.0, 1.0]


class TestStatAnomalyDetect:
    def test_spike_detected(self):
        state = _base_state(
            normalized_series={"A": [10, 10, 10, 40]},
            forecast={"A": [10.0]},
            confidence_interval={"A": {"lower": [6.0], "upper": [14.0]}},
        )
        out = StatAnomalyDetectNode().execute(state)
        assert out["status"] == AgentStatus.SUCCESS
        assert len(out["anomalies"]) == 1
        assert out["anomalies"][0]["anomaly_type"] == "demand_spike"
        assert out["anomalies"][0]["severity"] == "high"

    def test_within_band_no_anomaly(self):
        state = _base_state(
            normalized_series={"A": [10, 10, 10, 11]},
            forecast={"A": [10.0]},
            confidence_interval={"A": {"lower": [6.0], "upper": [14.0]}},
        )
        out = StatAnomalyDetectNode().execute(state)
        assert out["anomalies"] == []

    def test_demand_drop_is_seasonal_anomaly(self):
        # actual below lower band -> demand drop -> seasonal_anomaly (not reorder)
        state = _base_state(
            normalized_series={"A": [10, 10, 10, 2]},
            forecast={"A": [10.0]},
            confidence_interval={"A": {"lower": [6.0], "upper": [14.0]}},
        )
        out = StatAnomalyDetectNode().execute(state)
        assert out["anomalies"][0]["anomaly_type"] == "seasonal_anomaly"
        assert "再発注" not in out["anomalies"][0]["recommended_action"]

    def test_moderate_above_band_is_stockout_risk(self):
        # above upper but modest surge (|dev| < 0.5) -> stockout_risk
        state = _base_state(
            normalized_series={"A": [10, 10, 10, 13]},
            forecast={"A": [10.0]},
            confidence_interval={"A": {"lower": [8.0], "upper": [12.0]}},
        )
        out = StatAnomalyDetectNode().execute(state)
        assert out["anomalies"][0]["anomaly_type"] == "stockout_risk"


class TestEventAttribution:
    def test_promo_spike_attributed_expected(self):
        anomalies = [{"sku": "A", "anomaly_type": "demand_spike", "severity": "high"}]
        out = EventAttributionNode().execute(_base_state(anomalies=anomalies, covariate_matrix={"promo": [0, 0, 1]}))
        assert out["attributed_anomalies"][0]["attribution"] == "expected"

    def test_non_promo_spike_is_true_anomaly(self):
        anomalies = [{"sku": "A", "anomaly_type": "demand_spike", "severity": "high"}]
        out = EventAttributionNode().execute(_base_state(anomalies=anomalies, covariate_matrix={"promo": [0, 0, 0]}))
        assert out["attributed_anomalies"][0]["attribution"] == "true_anomaly"

    def test_negative_covariate_does_not_trip_promo(self):
        # price increase encoded as negative must NOT count as active promo
        anomalies = [{"sku": "A", "anomaly_type": "demand_spike"}]
        out = EventAttributionNode().execute(
            _base_state(anomalies=anomalies, covariate_matrix={"price_discount": [0, -0.05]})
        )
        assert out["attributed_anomalies"][0]["attribution"] == "true_anomaly"

    def test_seasonal_anomaly_not_promo_suppressed(self):
        # demand-down anomaly is never suppressed by promo
        anomalies = [{"sku": "A", "anomaly_type": "seasonal_anomaly"}]
        out = EventAttributionNode().execute(_base_state(anomalies=anomalies, covariate_matrix={"promo": [0, 1]}))
        assert out["attributed_anomalies"][0]["attribution"] == "true_anomaly"


class TestCovariateForecastRemoteWiring:
    """`remote` is the production path: the node builds an external HTTP caller;
    the agent holds no model. Verified without a real network call."""

    def test_resolve_service_builds_remote_caller_from_trusted_secret(self):
        from src.services.timesfm_service import RemoteTimesFMCaller

        node = CovariateForecastNode()
        # endpoint comes from the trusted secrets provider, NOT request input
        node._secret = lambda key: {
            "timesfm_endpoint": "https://timesfm.internal/forecast",
            "timesfm_api_token": "tok",
        }.get(key)
        svc = node._resolve_service({"inference_mode": "remote"})
        assert isinstance(svc._backend, RemoteTimesFMCaller)

    def test_remote_without_endpoint_secret_raises(self):
        node = CovariateForecastNode()
        node._secret = lambda key: None  # no trusted endpoint configured
        with pytest.raises(ValueError):
            node._resolve_service({"inference_mode": "remote"})

    def test_remote_rejects_non_https_endpoint_ssrf_guard(self):
        node = CovariateForecastNode()
        node._secret = lambda key: "http://attacker.evil/x" if key == "timesfm_endpoint" else None
        with pytest.raises(ValueError):
            node._resolve_service({"inference_mode": "remote"})

    def test_request_cannot_inject_endpoint(self):
        # Even if the request carries timesfm_endpoint, it is ignored (no secret -> raise)
        node = CovariateForecastNode()
        node._secret = lambda key: None
        with pytest.raises(ValueError):
            node._resolve_service({"inference_mode": "remote", "timesfm_endpoint": "https://attacker.evil/x"})

    def test_resolve_service_default_for_baseline(self):
        node = CovariateForecastNode()
        svc = node._resolve_service({"inference_mode": "baseline"})
        assert svc._backend is None  # resolves baseline by mode

    def test_execute_via_remote_endpoint(self):
        from src.services.timesfm_service import RemoteTimesFMCaller, TimesFMService

        def fake_post(url, payload, headers, timeout):
            return {
                "forecast": [10.0] * payload["horizon"],
                "confidence_interval": {"lower": [8.0] * payload["horizon"], "upper": [12.0] * payload["horizon"]},
            }

        node = CovariateForecastNode(
            service=TimesFMService(backend=RemoteTimesFMCaller("https://x/y", http_post=fake_post))
        )
        state = _base_state(
            parsed_request={"inference_mode": "remote", "horizon": 4},
            normalized_series={"A": [10, 11, 9, 10]},
            covariate_matrix={"promo": [0, 0, 1, 0]},
        )
        out = node.execute(state)
        assert out["status"] == AgentStatus.SUCCESS
        assert out["forecast"]["A"] == [10.0, 10.0, 10.0, 10.0]
        assert out["confidence_interval"]["A"]["upper"] == [12.0, 12.0, 12.0, 12.0]


class TestPreProcessSecurity:
    def test_s5_over_limit_rejected(self):
        import json

        req = {"series": {f"S{i}": [1, 2, 3] for i in range(600)}}
        state = _base_state(user_input=json.dumps(req))
        gated = PreProcessNode()._extra_security_gate_input(state)
        assert gated["status"] == AgentStatus.ERROR.value

    def test_within_limit_ok(self):
        import json

        req = {"series": {"A": [1, 2, 3]}}
        state = _base_state(user_input=json.dumps(req))
        gated = PreProcessNode()._extra_security_gate_input(state)
        assert gated.get("status") != AgentStatus.ERROR.value


class _ContentLLM(BaseLLM):
    """BaseLLM subclass returning caller-chosen content (adversarial replies)."""

    def __init__(self, content: str):
        self.calls: list[list] = []
        self._content = content

    def complete(self, messages: list) -> dict:
        self.calls.append(messages)
        return {"content": self._content, "tool_calls": [], "model": "fake", "usage": {}}

    def stream(self, messages: list):
        yield self.complete(messages)["content"]

    def bind_tools(self, tools: list) -> "_ContentLLM":
        return self


class TestReportGenerateNodeSecurityBoundary:
    """The narration node produces free text, so its own S-1/S-3 must hold.

    Driven through `node(state)` (i.e. BaseNode.__call__) so the framework gates
    actually run — calling execute() directly would bypass exactly what is under
    test.
    """

    _STATE = {
        "attributed_anomalies": [
            {
                "sku": "SKU-1001",
                "anomaly_type": "demand_spike",
                "severity": "high",
                "deviation": 3.0,
                "confidence_score": 1.0,
                "attribution": "true_anomaly",
                "recommended_action": "act",
            },
        ],
        "parsed_request": {"period": "2026-06", "inference_mode": "baseline"},
        "correlation_id": "report-node-security-test",
    }

    def _state(self, trust: TrustLevel) -> dict:
        return {**self._STATE, "caller_trust_level": trust.value}

    def test_pii_in_the_llm_reply_is_redacted_before_it_reaches_state(self):
        """S-3 belongs to the node that PRODUCES the text.

        The outer PostProcessNode redacts the same field, but only after the
        inner result has already travelled through State (and any checkpoint).
        """
        llm = _ContentLLM("# レポート\nSKU-1001 の担当は taro.yamada@example.com です。")
        node = ReportGenerateNode(llm=llm)
        result = node(self._state(TrustLevel.VERIFIED_EXTERNAL))
        assert llm.calls, "the LLM was not reached"
        assert "taro.yamada@example.com" not in result["report_markdown"]
        assert "REDACTED" in result["report_markdown"]

    def test_anonymous_caller_never_reaches_the_billable_llm_call(self):
        llm = _ContentLLM("# レポート\nSKU-1001")
        node = ReportGenerateNode(llm=llm)
        result = node(self._state(TrustLevel.ANONYMOUS))
        assert llm.calls == [], "S-1 must refuse before execute() spends an LLM call"
        assert "S-1 trust gate denied" in result["error_log"][0]
