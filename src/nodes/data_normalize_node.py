"""AgentCore Platform v1.0 — data_normalize (inner, deterministic Tool)."""

from __future__ import annotations

import json
from typing import ClassVar

from framework.nodes.function_node import FunctionNode, emit_trace_event
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel


class DataNormalizeNode(FunctionNode):
    """Normalize time series and assemble the covariate matrix.

    - Coerce series points to float, forward-fill missing (None) values.
    - Assemble covariates into aligned numeric columns.
    Deterministic: no model calls, fully reproducible.
    """

    # S-1 trust level is explicit by design, not inherited implicitly.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    @staticmethod
    def _clean_series(points: list) -> list[float]:
        cleaned: list[float] = []
        last = 0.0
        for p in points:
            if p is None:
                cleaned.append(last)  # forward-fill
                continue
            try:
                v = float(p)
            except (TypeError, ValueError):
                v = last
            cleaned.append(v)
            last = v
        return cleaned

    def execute(self, state: dict) -> dict:
        # Inner graph entry: the request arrives as a JSON string in user_input
        # (packed by the outer DemandAnomalyGraphNode.extract_input). Fall back to
        # an already-parsed parsed_request if present (direct inner-graph tests).
        parsed = state.get("parsed_request")
        if not parsed:
            raw = state.get("user_input", "")
            try:
                parsed = json.loads(raw) if isinstance(raw, str) and raw else {}
            except (json.JSONDecodeError, ValueError):
                return {
                    "status": AgentStatus.ERROR.value,
                    "error_log": ["DataNormalizeNode: inner user_input is not valid JSON"],
                }

        series = parsed.get("series") or {}
        target_skus = parsed.get("target_skus") or list(series.keys())

        normalized = {sku: self._clean_series(series.get(sku, [])) for sku in target_skus}

        covariate_matrix: dict[str, list[float]] = {}
        for name, values in (parsed.get("covariates") or {}).items():
            covariate_matrix[name] = self._clean_series(values if isinstance(values, list) else [])

        emit_trace_event(
            "data_normalized",
            {"sku_count": len(normalized), "covariate_count": len(covariate_matrix)},
            state,
        )
        return {
            "parsed_request": parsed,
            "normalized_series": normalized,
            "covariate_matrix": covariate_matrix,
            "status": AgentStatus.SUCCESS.value,
        }
