"""AgentCore Platform v1.0 — PostProcessNode (OutputGate + S-3)."""

from __future__ import annotations

from framework.nodes.function_node import FunctionNode, detect_pii, emit_trace_event
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel


class PostProcessNode(FunctionNode):
    """OutputGate: shape the structured JSON output + reports, enforce S-3.

    Reads `anomaly_result` (bundled by DemandAnomalyGraphNode.merge_output):
        {attributed_anomalies, report_markdown, report_csv}
    Produces `formatted_output` = structured JSON + report/CSV.
    """

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict) -> dict:
        result = state.get("anomaly_result") or {}
        attributed = result.get("attributed_anomalies", [])
        true_anomalies = [a for a in attributed if a.get("attribution") == "true_anomaly"]

        formatted = {
            "anomaly_detected": bool(true_anomalies),
            "anomaly_count": len(true_anomalies),
            "anomalies": [
                {
                    "affected_skus": [a.get("sku")],
                    "anomaly_type": a.get("anomaly_type"),
                    "severity": a.get("severity"),
                    "recommended_action": a.get("recommended_action"),
                    "confidence_score": a.get("confidence_score"),
                }
                for a in true_anomalies
            ],
            "report_markdown": result.get("report_markdown", ""),
            "report_csv": result.get("report_csv", ""),
            # Which narration path actually ran. Declared generation_mode is "llm",
            # so a deterministic fallback must be visible to the caller, never silent.
            "narration_mode": result.get("narration_mode", ""),
        }
        # Only present when the LLM narration path was wired but did not deliver
        # (empty reply / provider failure). Absent = nothing degraded.
        if result.get("degradation_reason"):
            formatted["degradation_reason"] = result["degradation_reason"]
        emit_trace_event(
            "output_gated",
            {"anomaly_detected": formatted["anomaly_detected"], "count": len(true_anomalies)},
            state,
        )
        return {
            "formatted_output": formatted,
            "status": AgentStatus.SUCCESS.value,
        }

    def _extra_security_gate_output(self, result: dict) -> dict:
        """S-3: re-scan output for PII (store/customer identifiers) and strip.

        Runs after the default credential scan. Removes any PII findings from
        the human-readable report bodies; the structured JSON carries SKU-level
        data only (no personal data by design).
        """
        formatted = result.get("formatted_output")
        if isinstance(formatted, dict):
            for key in ("report_markdown", "report_csv"):
                text = formatted.get(key, "")
                if isinstance(text, str) and text and detect_pii(text):
                    formatted[key] = "[REDACTED: PII detected in report body — review input data]"
        return result
