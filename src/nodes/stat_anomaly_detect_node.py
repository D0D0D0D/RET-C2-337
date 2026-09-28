"""AgentCore Platform v1.0 — stat_anomaly_detect (inner, DETERMINISTIC Tool).

The core detection step. NO LLM. Compares the latest actual against the
forecast confidence interval and classifies the deviation by rule. This is
what makes the agent's judgments reproducible and auditable (Issue A guardrail).
"""

from __future__ import annotations

from typing import ClassVar

from framework.nodes.function_node import FunctionNode, emit_trace_event
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel


def _severity(deviation_ratio: float) -> str:
    a = abs(deviation_ratio)
    if a >= 0.5:
        return "high"
    if a >= 0.2:
        return "medium"
    return "low"


class StatAnomalyDetectNode(FunctionNode):
    """Rule-based anomaly detection: actual vs forecast CI.

    Classification by deviation direction AND magnitude (all three types
    reachable):
      - actual > upper, large surge (|dev| >= 0.5) -> demand_spike
            (sudden surge -> escalate / expedite supply)
      - actual > upper, moderate                   -> stockout_risk
            (sustained demand above plan -> reorder to avoid running out)
      - actual < lower                             -> seasonal_anomaly
            (demand below plan -> oversupply / markdown to avoid food waste)
      - within band                                -> no anomaly
    """

    # S-1 trust level is explicit by design, not inherited implicitly.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: dict) -> dict:
        normalized = state.get("normalized_series") or {}
        forecast = state.get("forecast") or {}
        ci = state.get("confidence_interval") or {}

        anomalies: list[dict] = []
        for sku, series in normalized.items():
            if not series or sku not in forecast or not forecast[sku]:
                continue
            actual = series[-1]
            point = forecast[sku][0]
            band = ci.get(sku, {})
            lower = band.get("lower", [point])[0]
            upper = band.get("upper", [point])[0]
            denom = abs(point) if point else 1.0
            deviation_ratio = (actual - point) / denom

            if actual > upper:
                # demand above forecast
                if abs(deviation_ratio) >= 0.5:
                    atype = "demand_spike"
                    action = "カテゴリーマネージャーへエスカレーション・補充前倒し検討"
                else:
                    atype = "stockout_risk"
                    action = "再発注トリガー・在庫補充の検討（欠品リスク）"
            elif actual < lower:
                # demand below forecast -> oversupply / seasonal deviation
                atype = "seasonal_anomaly"
                action = "値下げ・需要見直しの検討（過剰在庫・食品ロス回避）"
            else:
                continue  # within CI — not an anomaly

            confidence = round(min(1.0, abs(deviation_ratio)), 4)
            anomalies.append(
                {
                    "sku": sku,
                    "anomaly_type": atype,
                    "severity": _severity(deviation_ratio),
                    "deviation": round(deviation_ratio, 4),
                    "confidence_score": confidence,
                    "recommended_action": action,
                }
            )

        emit_trace_event("anomalies_detected", {"count": len(anomalies)}, state)
        return {"anomalies": anomalies, "status": AgentStatus.SUCCESS.value}
