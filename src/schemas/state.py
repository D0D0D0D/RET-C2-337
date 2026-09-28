"""AgentCore Platform v1.0 — RET-C2-337 State schema."""

# ADR-005: State must be a flat TypedDict — never Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects
# cause silent corruption.  Extend AgentState with agent-specific
# fields only.  Do NOT add credentials, secrets, or Pydantic models.

from __future__ import annotations

from framework.schemas.agent_state import AgentState


class State(AgentState):
    """Demand-anomaly agent state.

    Shared fields (user_input, status, session_id, node_history, error_log,
    validated_input, formatted_output, hitl_*, etc.) are inherited from
    AgentState. Only domain-specific, msgpack-safe fields are added below.
    """

    # pre_process (InputParse)
    parsed_request: dict          # {series, covariates, target_skus, period, inference_mode}

    # inner: data_normalize
    normalized_series: dict       # {sku: [points]}
    covariate_matrix: dict        # {covariate_name: [values]}

    # inner: covariate_forecast
    forecast: dict                # {sku: [point_forecast]}
    confidence_interval: dict     # {sku: {"lower": [...], "upper": [...]}}

    # inner: stat_anomaly_detect
    anomalies: list               # [{sku, anomaly_type, severity, deviation, confidence_score}]

    # inner: event_attribution
    attributed_anomalies: list    # anomalies + {"attribution": "expected"|"true_anomaly"}

    # inner: report_generate
    report_markdown: str
    report_csv: str
    report_narration_mode: str      # "llm" | "deterministic" | "custom"
    report_degradation_reason: str  # "" when the narration path ran as declared

    # inner -> outer (merge_output)
    anomaly_result: dict          # {attributed_anomalies, report_markdown, report_csv}
