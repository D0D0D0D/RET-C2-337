# RET-C2-337 — Integration tests: full agent invoke + F-1 false-positive measurement

import json

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from shared.services.llm.base_llm import BaseLLM

from src.graph.graph import RETDemandAnomalyAgent


def _ctx():
    return InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL, caller_id="itest")


def _agent():
    a = RETDemandAnomalyAgent()
    a.compile()
    return a


def _invoke(agent, request):
    return agent.invoke(json.dumps(request), ctx=_ctx())


class TestEndToEnd:
    def test_happy_path_detects_spike(self):
        agent = _agent()
        req = {
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
        out = _invoke(agent, req)["output"]
        assert out["anomaly_detected"] is True
        assert out["anomaly_count"] == 1
        assert out["anomalies"][0]["affected_skus"] == ["SKU2"]
        assert out["report_markdown"]
        assert out["report_csv"].splitlines()[0].startswith("sku,")

    def test_node_history_backbone(self):
        agent = _agent()
        out = _invoke(agent, {"series": {"A": [10] * 14}, "covariates": {}})
        # full 5-node outer backbone ran
        names = [n.get("node") if isinstance(n, dict) else n for n in out["node_history"]]
        assert any("Initialize" in str(n) for n in names)
        assert any("Finalize" in str(n) for n in names)

    def test_s5_over_limit_rejected(self):
        agent = _agent()
        req = {"series": {f"S{i}": [1, 2, 3] for i in range(600)}}
        req["target_skus"] = list(req["series"])
        out = _invoke(agent, req)
        assert str(out["status"]).endswith("ERROR") or out["status"] == "error"

    def test_malformed_input_rejected(self):
        agent = _agent()
        out = agent.invoke("not a json request", ctx=_ctx())
        assert str(out["status"]).endswith("ERROR") or out["status"] == "error"


class TestF1FalsePositiveSuppression:
    """F-1 (Conditional-Go validation): measure false-positive suppression for
    promo-driven spikes. A promo-driven spike must be attributed 'expected'
    (suppressed), not flagged as a true anomaly. We require the suppression rate
    to be high on synthetic promo data and the false-positive rate to be low.
    """

    @staticmethod
    def _series_with_spike(n_history=13, spike=40, base=10):
        return [base, base + 1, base - 1, base, base + 2, base, base + 1][:n_history] * 2

    def test_promo_spikes_are_suppressed(self):
        agent = _agent()
        n = 20
        # Each SKU has a final-point spike that is ENTIRELY promo-driven.
        series = {}
        for i in range(n):
            s = [10, 11, 9, 10, 12, 10, 11, 10, 11, 9, 10, 12, 10, 40]
            series[f"SKU{i}"] = s
        # promo active on the final point -> spike is expected
        req = {
            "series": series,
            "covariates": {"promo": [0] * 13 + [1]},
            "target_skus": list(series),
            "inference_mode": "baseline",
        }
        out = _invoke(agent, req)["output"]
        false_positives = out["anomaly_count"]  # promo spikes wrongly flagged
        fp_rate = false_positives / n
        assert fp_rate <= 0.10, f"F-1: promo-driven false-positive rate too high: {fp_rate:.2%}"

    def test_true_spikes_still_detected_without_promo(self):
        agent = _agent()
        n = 20
        series = {f"SKU{i}": [10, 11, 9, 10, 12, 10, 11, 10, 11, 9, 10, 12, 10, 40] for i in range(n)}
        req = {
            "series": series,
            "covariates": {"promo": [0] * 14},  # no promo -> genuine anomalies
            "target_skus": list(series),
            "inference_mode": "baseline",
        }
        out = _invoke(agent, req)["output"]
        # recall: genuine spikes must be caught
        detection_rate = out["anomaly_count"] / n
        assert detection_rate >= 0.90, f"recall too low without promo: {detection_rate:.2%}"


class _RecordingLLM(BaseLLM):
    """Real BaseLLM subclass injected the way production does: config["llm"]."""

    def __init__(self, content="# LLM が書いたレポート\nSKU2 の需要急増を確認。"):
        self.calls: list[list] = []
        self._content = content

    def complete(self, messages: list) -> dict:
        self.calls.append(messages)
        return {"content": self._content, "tool_calls": [], "model": "fake", "usage": {}}

    def stream(self, messages: list):
        yield self.complete(messages)["content"]

    def bind_tools(self, tools: list) -> "_RecordingLLM":
        return self


class TestLLMGenerationModeIsReal:
    """config/agent.yaml declares generation_mode: "llm".

    The declaration is only true if an LLM injected at the OUTER graph reaches
    report_generate in the INNER workflow. This exercises the whole invoke path
    (not the node directly), because the hand-off under test is
    RETDemandAnomalyAgent.register_nodes -> DemandAnomalyGraphNode._parent_config
    -> DemandAnomalyWorkflowGraph.register_nodes.
    """

    _REQ = {
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

    def test_injected_llm_reaches_report_generate_through_the_subgraph(self):
        llm = _RecordingLLM()
        agent = RETDemandAnomalyAgent(config={"llm": llm})
        agent.compile()
        out = _invoke(agent, self._REQ)["output"]

        assert llm.calls, "config['llm'] never reached the inner report_generate node"
        assert out["narration_mode"] == "llm"
        assert out["report_markdown"] == "# LLM が書いたレポート\nSKU2 の需要急増を確認。"
        # Detection stays deterministic and authoritative even though narration is LLM-written.
        assert out["anomaly_count"] == 1
        assert out["anomalies"][0]["affected_skus"] == ["SKU2"]
        assert out["report_csv"].splitlines()[0].startswith("sku,")
        assert "degradation_reason" not in out

    def test_prompt_carries_the_detection_result_not_the_raw_series(self):
        llm = _RecordingLLM()
        agent = RETDemandAnomalyAgent(config={"llm": llm})
        agent.compile()
        _invoke(agent, self._REQ)
        user = llm.calls[0][1]["content"]
        assert "true_anomaly" in user and "SKU2" in user
        # The LLM narrates the verdict; it never re-runs detection on raw history.
        assert '"series"' not in user

    def test_without_an_llm_the_fallback_is_declared_not_silent(self):
        agent = _agent()
        out = _invoke(agent, self._REQ)["output"]
        assert out["narration_mode"] == "deterministic"
        assert out["report_markdown"]

    def test_provider_failure_surfaces_as_degradation_through_the_whole_invoke(self):
        """A narration failure must reach the caller, not vanish inside the subgraph."""

        class _FailingLLM(_RecordingLLM):
            def complete(self, messages: list) -> dict:
                self.calls.append(messages)
                raise RuntimeError("provider exploded: token=should-not-surface")

        llm = _FailingLLM()
        agent = RETDemandAnomalyAgent(config={"llm": llm})
        agent.compile()
        out = _invoke(agent, self._REQ)["output"]

        assert llm.calls, "the LLM was not reached"
        assert out["degradation_reason"] == "llm_narration_failed:RuntimeError"
        assert "should-not-surface" not in json.dumps(out, ensure_ascii=False)
        assert out["narration_mode"] == "deterministic"
        # Detection is unaffected: a narration failure never changes the verdict.
        assert out["anomaly_count"] == 1
        assert out["report_markdown"]

    def test_system_prompt_from_config_reaches_the_llm(self):
        """`system_prompt` is in _INNER_CONFIG_KEYS — a forwarded key that is
        never delivered is dead config, so the whole chain is asserted here."""
        llm = _RecordingLLM()
        agent = RETDemandAnomalyAgent(config={"llm": llm, "system_prompt": "CUSTOM-SYSTEM"})
        agent.compile()
        _invoke(agent, self._REQ)
        assert llm.calls[0][0]["content"] == "CUSTOM-SYSTEM"
