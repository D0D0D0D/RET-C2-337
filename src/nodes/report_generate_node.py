"""AgentCore Platform v1.0 — report_generate (inner, LLM step).

The ONLY LLM step. Narrates the already-computed, auditable detection result
into a Markdown report; the CSV stays deterministic. Detection is never
re-decided here. The LLM arrives via graph config (`config["llm"]`, forwarded
from the outer graph); without one the node falls back to deterministic
narration and reports that fact through `report_narration_mode` so the fallback
is visible to the caller instead of silent.
"""

from __future__ import annotations

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode, detect_pii, emit_trace_event
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.services.llm_client import LLMReportClient


class ReportGenerateNode(FunctionNode):
    """Generate Markdown report + CSV from the attributed anomalies."""

    # S-1: explicit by design, not inherited implicitly.
    # The node now issues a billable external LLM call, so it must not be
    # reachable by an anonymous caller; the outer entry contract
    # (config/agent.yaml, PostProcessNode) is VERIFIED_EXTERNAL already.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(
        self,
        client: LLMReportClient | None = None,
        llm: Any | None = None,
        system_prompt: str | None = None,
    ) -> None:
        super().__init__()
        # Static configuration only — no mutable per-invocation state (node
        # instances are shared across every invocation).
        self._client = client or LLMReportClient(llm=llm, system_prompt=system_prompt)

    def execute(self, state: dict) -> dict:
        attributed = state.get("attributed_anomalies") or []
        parsed = state.get("parsed_request") or {}
        meta = {
            "period": parsed.get("period", ""),
            "inference_mode": parsed.get("inference_mode", "baseline"),
        }
        report_markdown, narration_mode, degradation_reason = self._client.render_report(
            attributed, meta
        )
        report_csv = self._client.render_csv(attributed)

        emit_trace_event(
            "report_generated",
            {
                "markdown_chars": len(report_markdown),
                "csv_rows": report_csv.count("\n"),
                "narration_mode": narration_mode,
                "degradation_reason": degradation_reason,
            },
            state,
        )
        return {
            "report_markdown": report_markdown,
            "report_csv": report_csv,
            "report_narration_mode": narration_mode,
            "report_degradation_reason": degradation_reason,
            "status": AgentStatus.SUCCESS.value,
        }

    def _extra_security_gate_output(self, result: dict) -> dict:
        """S-3: redact PII from the narration BEFORE it is written to State.

        The outer PostProcessNode redacts the same fields, but only after the
        inner result has already travelled through State (and, when a
        checkpointer is configured, through the checkpoint). Redaction belongs
        to the node that puts the content into its return dict — the LLM reply
        is free text this node produces, so it is gated here as well.
        """
        text = result.get("report_markdown", "")
        if isinstance(text, str) and text and detect_pii(text):
            result["report_markdown"] = (
                "[REDACTED: PII detected in report body — review input data]"
            )
        return result
