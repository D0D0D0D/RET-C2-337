# RET-C2-337 — Unit Tests: services (deterministic forecaster + LLM seam)

import pytest

from src.services.timesfm_service import RemoteTimesFMCaller, TimesFMService, baseline_forecast
from shared.services.llm.base_llm import BaseLLM

from src.services.llm_client import LLMReportClient, anomalies_to_csv


class TestBaselineForecast:
    def test_deterministic_reproducible(self):
        series = [10, 11, 9, 10, 12, 10, 11] * 3
        a = baseline_forecast(series, {}, 7)
        b = baseline_forecast(series, {}, 7)
        assert a == b  # no randomness — reproducibility is the moat
        point, ci = a
        assert len(point) == 7
        assert len(ci["lower"]) == 7 and len(ci["upper"]) == 7
        for lo, p, hi in zip(ci["lower"], point, ci["upper"]):
            assert lo <= p <= hi

    def test_covariate_lifts_forecast(self):
        series = [10] * 14
        no_cov, _ = baseline_forecast(series, {}, 3)
        with_promo, _ = baseline_forecast(series, {"promo": [0] * 13 + [5]}, 3)
        assert with_promo[0] > no_cov[0]  # promo covariate pushes demand up

    def test_empty_series(self):
        point, ci = baseline_forecast([], {}, 4)
        assert point == [0.0, 0.0, 0.0, 0.0]


class TestTimesFMService:
    def test_forecast_per_sku(self):
        svc = TimesFMService()
        out = svc.forecast({"A": [10] * 14, "B": [5] * 14}, {}, 7, inference_mode="baseline")
        assert set(out["forecast"]) == {"A", "B"}
        assert "lower" in out["confidence_interval"]["A"]

    def test_injectable_backend(self):
        called = {}

        def fake_backend(series, cov, horizon):
            called["hit"] = True
            return [42.0] * horizon, {"lower": [0.0] * horizon, "upper": [99.0] * horizon}

        svc = TimesFMService(backend=fake_backend)
        out = svc.forecast({"A": [1, 2, 3]}, {}, 2)
        assert called.get("hit") is True
        assert out["forecast"]["A"] == [42.0, 42.0]


class TestRemoteTimesFMCaller:
    """PRODUCTION backend: calls an EXTERNAL endpoint. Fully testable via an
    injected http_post (no real network) — the production path runs in CI."""

    def test_posts_payload_and_parses_response(self):
        captured = {}

        def fake_post(url, payload, headers, timeout):
            captured["url"] = url
            captured["payload"] = payload
            captured["headers"] = headers
            return {
                "forecast": [10.0, 11.0, 12.0],
                "confidence_interval": {"lower": [9.0, 10.0, 11.0], "upper": [11.0, 12.0, 13.0]},
            }

        caller = RemoteTimesFMCaller("https://timesfm.example/forecast", auth_token="tok", http_post=fake_post)
        point, ci = caller([10, 11, 9, 10], {"promo": [0, 0, 1, 0]}, horizon=3)

        assert captured["url"] == "https://timesfm.example/forecast"
        assert captured["payload"]["horizon"] == 3
        assert captured["payload"]["series"] == [10.0, 11.0, 9.0, 10.0]
        assert captured["headers"]["Authorization"] == "Bearer tok"
        assert point == [10.0, 11.0, 12.0]
        assert ci["lower"] == [9.0, 10.0, 11.0] and ci["upper"] == [11.0, 12.0, 13.0]

    def test_empty_endpoint_rejected(self):
        with pytest.raises(ValueError):
            RemoteTimesFMCaller("")

    def test_no_auth_header_when_token_absent(self):
        seen = {}

        def fake_post(url, payload, headers, timeout):
            seen["headers"] = headers
            return {"forecast": [1.0]}

        RemoteTimesFMCaller("https://x/y", http_post=fake_post)([1, 2, 3], {}, 1)
        assert "Authorization" not in seen["headers"]

    def test_service_uses_injected_remote_caller(self):
        def fake_post(url, payload, headers, timeout):
            return {"forecast": [5.0, 5.0], "confidence_interval": {"lower": [4.0, 4.0], "upper": [6.0, 6.0]}}

        svc = TimesFMService(backend=RemoteTimesFMCaller("https://x/y", http_post=fake_post))
        out = svc.forecast({"A": [1, 2, 3]}, {}, horizon=2, inference_mode="remote")
        assert out["forecast"]["A"] == [5.0, 5.0]
        assert out["confidence_interval"]["A"]["upper"] == [6.0, 6.0]


class TestLLMReportClient:
    def test_markdown_lists_true_anomalies_only(self):
        anomalies = [
            {
                "sku": "A",
                "anomaly_type": "demand_spike",
                "severity": "high",
                "deviation": 1.0,
                "confidence_score": 0.9,
                "attribution": "true_anomaly",
                "recommended_action": "act",
            },
            {"sku": "B", "anomaly_type": "demand_spike", "severity": "low", "attribution": "expected"},
        ]
        md = LLMReportClient().render_markdown(anomalies, {"period": "P", "inference_mode": "baseline"})
        assert "A" in md and "demand_spike" in md
        assert "プロモ起因として除外: 1" in md

    def test_csv_header_and_no_pii(self):
        csv = anomalies_to_csv([{"sku": "A", "anomaly_type": "demand_spike"}])
        assert csv.splitlines()[0].startswith("sku,anomaly_type,severity")


class _RecordingLLM(BaseLLM):
    """Real BaseLLM subclass — an object that only quacks would hide a contract
    violation (a non-BaseLLM fake cannot prove the production seam accepts a
    provider client)."""

    def __init__(self, content="LLM が書いた本文", raises=None):
        self.calls: list[list] = []
        self._content = content
        self._raises = raises

    def complete(self, messages: list) -> dict:
        self.calls.append(messages)
        if self._raises is not None:
            raise self._raises
        return {"content": self._content, "tool_calls": [], "model": "fake", "usage": {}}

    def stream(self, messages: list):
        yield self.complete(messages)["content"]

    def bind_tools(self, tools: list) -> "_RecordingLLM":
        return self


class TestLLMNarrationSeam:
    """`generation_mode: llm` — report_generate must really reach the injected LLM."""

    _ANOMALIES = [
        {
            "sku": "SKU-1001",
            "anomaly_type": "demand_spike",
            "severity": "high",
            "deviation": 1.0,
            "confidence_score": 0.9,
            "attribution": "true_anomaly",
            "recommended_action": "act",
        },
    ]
    _META = {"period": "2026-06", "inference_mode": "baseline"}

    def test_injected_llm_writes_the_markdown(self):
        llm = _RecordingLLM(content="# LLM レポート\nSKU-1001 の需要急増を確認。")
        md, mode, reason = LLMReportClient(llm=llm).render_report(self._ANOMALIES, self._META)
        assert md == "# LLM レポート\nSKU-1001 の需要急増を確認。"
        assert mode == "llm" and reason == ""
        assert len(llm.calls) == 1
        # The prompt is built in src/, carries the detection result, and forbids
        # re-deciding detection.
        system, user = llm.calls[0][0]["content"], llm.calls[0][1]["content"]
        assert "異常判定をやり直さない" in system
        assert "demand_spike" in user and "2026-06" in user

    def test_csv_is_never_narrated(self):
        llm = _RecordingLLM(content="# LLM レポート SKU-1001")
        client = LLMReportClient(llm=llm)
        client.render_report(self._ANOMALIES, self._META)
        csv = client.render_csv(self._ANOMALIES)
        assert csv.splitlines()[0].startswith("sku,anomaly_type,severity")
        assert "LLM レポート" not in csv

    def test_no_llm_falls_back_deterministically_and_says_so(self):
        md, mode, reason = LLMReportClient().render_report(self._ANOMALIES, self._META)
        assert "需要異常検知レポート" in md
        assert mode == "deterministic" and reason == ""

    def test_provider_failure_degrades_without_leaking_the_message(self):
        secret = "api_key=sk-live-should-never-surface"
        llm = _RecordingLLM(raises=RuntimeError(secret))
        md, mode, reason = LLMReportClient(llm=llm).render_report(self._ANOMALIES, self._META)
        assert "需要異常検知レポート" in md  # detection result still reported
        assert mode == "deterministic"
        assert reason == "llm_narration_failed:RuntimeError"
        assert secret not in reason  # provider text never reaches State

    def test_empty_reply_is_a_degradation_not_an_empty_report(self):
        llm = _RecordingLLM(content="   ")
        md, mode, reason = LLMReportClient(llm=llm).render_report(self._ANOMALIES, self._META)
        assert "需要異常検知レポート" in md
        assert mode == "deterministic" and reason == "llm_narration_empty"

    def test_narration_that_drops_a_detected_anomaly_is_refused(self):
        """Grounding guard: prose that summarises a true anomaly away is not published."""
        two = self._ANOMALIES + [
            {
                "sku": "SKU-2002",
                "anomaly_type": "demand_drop",
                "severity": "medium",
                "deviation": -0.8,
                "confidence_score": 0.8,
                "attribution": "true_anomaly",
                "recommended_action": "check",
            },
        ]
        llm = _RecordingLLM(content="# レポート\nSKU-1001 のみ言及し、もう一方を落とした本文")
        md, mode, reason = LLMReportClient(llm=llm).render_report(two, self._META)
        assert mode == "deterministic" and reason == "llm_narration_incomplete"
        assert "SKU-2002" in md and "demand_drop" in md  # deterministic table is complete

    def test_expected_only_skus_are_not_required_in_the_narration(self):
        """A promo-suppressed anomaly must NOT force a mention (no false degradation)."""
        mixed = self._ANOMALIES + [
            {"sku": "ZZZ", "anomaly_type": "demand_spike", "severity": "low", "attribution": "expected"},
        ]
        llm = _RecordingLLM(content="# レポート\nSKU-1001 のみ要対応。")
        md, mode, reason = LLMReportClient(llm=llm).render_report(mixed, self._META)
        assert mode == "llm" and reason == ""

    def test_custom_system_prompt_replaces_the_default(self):
        llm = _RecordingLLM(content="SKU-1001")
        LLMReportClient(llm=llm, system_prompt="CUSTOM").render_report(self._ANOMALIES, self._META)
        assert llm.calls[0][0]["content"] == "CUSTOM"

    def test_sku_id_is_matched_on_token_boundaries(self):
        """ "SKU-1001" must not be considered mentioned by "SKU-10012"."""
        llm = _RecordingLLM(content="# レポート\nSKU-10012 の話しかしていない本文")
        _, mode, reason = LLMReportClient(llm=llm).render_report(self._ANOMALIES, self._META)
        assert mode == "deterministic" and reason == "llm_narration_incomplete"

    def test_japanese_particle_directly_after_the_sku_still_counts(self):
        """ "SKU-1001の需要" — a \\b-based check would wrongly degrade this."""
        llm = _RecordingLLM(content="# レポート\nSKU-1001の需要が急増しました。")
        _, mode, reason = LLMReportClient(llm=llm).render_report(self._ANOMALIES, self._META)
        assert mode == "llm" and reason == ""
