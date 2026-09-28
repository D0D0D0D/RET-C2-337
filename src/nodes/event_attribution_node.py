"""AgentCore Platform v1.0 — event_attribution (inner, RULE Tool).

Separates "expected" variation (promo/holiday-driven) from "true" anomalies by
matching detected anomalies against a known-event KB. This is the false-positive
suppression step — the heart of the template's moat (covariate + event know-how).
NO LLM (deterministic, auditable).
"""

from __future__ import annotations

from typing import ClassVar

from framework.nodes.function_node import FunctionNode, emit_trace_event
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel


class EventAttributionNode(FunctionNode):
    """Attribute each anomaly to 'expected' (promo-driven) or 'true_anomaly'.

    Heuristic: if a demand-up anomaly (demand_spike / stockout_risk) coincides
    with an ACTIVE promo/holiday covariate signal (last value strictly > 0), it
    is expected (planned) variation, not a true anomaly. Customer event/promo
    calendars feed the covariate matrix (design dependency F-2). Demand-down
    anomalies (seasonal_anomaly) are never promo-suppressed. The rule is
    conservative and reproducible; refine with the F-1 false-positive measurement.
    """

    # S-1 trust level is explicit by design, not inherited implicitly.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    PROMO_SIGNALS = ("promo", "holiday", "price_discount")
    # demand-up anomaly types a promo can legitimately explain
    DEMAND_UP_TYPES = ("demand_spike", "stockout_risk")

    @staticmethod
    def _last_positive(values) -> bool:
        if not values:
            return False
        try:
            return float(values[-1]) > 0
        except (TypeError, ValueError):
            return False

    def execute(self, state: dict) -> dict:
        anomalies = state.get("anomalies") or []
        covariate_matrix = state.get("covariate_matrix") or {}

        promo_active = any(
            self._last_positive(covariate_matrix.get(sig)) for sig in self.PROMO_SIGNALS
        )

        attributed: list[dict] = []
        expected_count = 0
        for a in anomalies:
            item = dict(a)
            if a.get("anomaly_type") in self.DEMAND_UP_TYPES and promo_active:
                item["attribution"] = "expected"
                expected_count += 1
            else:
                item["attribution"] = "true_anomaly"
            attributed.append(item)

        emit_trace_event(
            "anomalies_attributed",
            {"expected": expected_count, "true_anomaly": len(attributed) - expected_count},
            state,
        )
        return {"attributed_anomalies": attributed, "status": AgentStatus.SUCCESS.value}
