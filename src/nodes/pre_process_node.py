"""AgentCore Platform v1.0 — PreProcessNode (InputParse + S-2 / S-5)."""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict (changed keys only)
#  - Return AgentStatus enum constants — never plain strings
#  - S-2/S-5 input gate via _extra_security_gate_input() (do NOT override
#    _security_gate_input — it is @final and runs the default PII scan first)

from __future__ import annotations

import json

from framework.nodes.function_node import FunctionNode, emit_trace_event
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

# S-5: cap SKUs per invocation (rate/resource limit). Mirrors config/agent.yaml.
DEFAULT_MAX_SKUS_PER_CALL = 500


class PreProcessNode(FunctionNode):
    """InputParse: parse and validate the demand-anomaly request.

    Accepts the request as a JSON string in user_input, e.g.
        {"series": {"SKU1": [..]}, "covariates": {"promo": [..]},
         "target_skus": ["SKU1"], "period": "2026-06-01..2026-06-30",
         "inference_mode": "baseline"}
    """

    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict) -> dict:
        raw = state.get("validated_input") or state.get("user_input", "")
        try:
            request = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["PreProcessNode: user_input is not valid JSON request"],
            }

        series = request.get("series") or {}
        if not isinstance(series, dict) or not series:
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": ["PreProcessNode: 'series' is required and must be a non-empty object"],
            }

        target_skus = request.get("target_skus") or list(series.keys())
        parsed = {
            "series": series,
            "covariates": request.get("covariates") or {},
            "target_skus": target_skus,
            "period": request.get("period", ""),
            "inference_mode": request.get("inference_mode", "baseline"),
            "horizon": int(request.get("horizon", 7)),
        }
        emit_trace_event(
            "input_parsed",
            {"sku_count": len(target_skus), "inference_mode": parsed["inference_mode"]},
            state,
        )
        return {
            "validated_input": raw if isinstance(raw, str) else json.dumps(raw),
            "parsed_request": parsed,
            "status": AgentStatus.SUCCESS.value,
        }

    def _extra_security_gate_input(self, state: dict) -> dict:
        """S-2/S-5: enforce SKU-count ceiling and input sanity.

        Runs as the input gate BEFORE execute(), so parsed_request is not set
        yet — parse the raw request here to count SKUs. To reject, set
        status=ERROR.value and append to error_log (must not raise).
        """
        raw = state.get("validated_input") or state.get("user_input", "")
        try:
            request = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            return state  # malformed JSON is surfaced as ERROR by execute()

        series = request.get("series") or {}
        target_skus = request.get("target_skus") or (list(series.keys()) if isinstance(series, dict) else [])
        if len(target_skus) > DEFAULT_MAX_SKUS_PER_CALL:
            state["status"] = AgentStatus.ERROR.value
            state.setdefault("error_log", []).append(
                f"S-5: target_skus ({len(target_skus)}) exceeds max_skus_per_call "
                f"({DEFAULT_MAX_SKUS_PER_CALL}); paginate the request upstream."
            )
        return state
