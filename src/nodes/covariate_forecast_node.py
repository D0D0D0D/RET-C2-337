"""AgentCore Platform v1.0 — covariate_forecast (inner, TimesFM Tool)."""

from __future__ import annotations

from typing import ClassVar

from framework.nodes.function_node import FunctionNode, emit_trace_event
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.services.timesfm_service import RemoteTimesFMCaller, TimesFMService


class CovariateForecastNode(FunctionNode):
    """Call TimesFM (per SKU) to get point forecast + confidence interval.

    Detection-side Tool: produces the forecast that stat_anomaly_detect compares
    actuals against. TimesFM compute is an EXTERNAL resource — in production this
    node calls a remote serving endpoint (`remote`) or BigQuery ML (`bigquery`);
    the agent process holds no model. `baseline` (in-process deterministic) is the
    default for CI/demo; `local` is dev-only. The TimesFMService is injectable so
    tests pin a deterministic/fake backend.
    """

    # S-1 trust level is explicit by design, not inherited implicitly.
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def __init__(self, service: TimesFMService | None = None) -> None:
        super().__init__()
        # If injected, the service is used for every mode (tests pin a fake).
        self._service = service

    def _resolve_service(self, parsed: dict) -> TimesFMService:
        """Pick the service for this request's inference_mode.

        For `remote`, build a RemoteTimesFMCaller that calls an EXTERNAL endpoint.
        SECURITY: the endpoint URL and auth token come ONLY from the trusted
        framework secrets provider (deploy-time config) — NEVER from request
        input — and the URL must be https. This prevents an attacker-supplied
        request from redirecting the call (SSRF) or exfiltrating the bearer
        token. For other modes, the default service resolves the backend by
        mode. An injected service always wins (test seam).
        """
        if self._service is not None:
            return self._service

        mode = parsed.get("inference_mode", "baseline")
        if mode == "remote":
            endpoint = self._secret("timesfm_endpoint")
            if not endpoint:
                raise ValueError(
                    "inference_mode='remote' requires a trusted 'timesfm_endpoint' "
                    "secret (deploy-time config). It must NOT come from request input."
                )
            self._validate_endpoint(endpoint)
            token = self._secret("timesfm_api_token")
            return TimesFMService(backend=RemoteTimesFMCaller(endpoint, auth_token=token))
        return TimesFMService()

    @staticmethod
    def _validate_endpoint(url: str) -> None:
        """Reject non-https endpoints (plaintext token leak) and malformed URLs."""
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError(f"timesfm_endpoint must be an https URL with a host (got: {url!r})")

    def _secret(self, key: str) -> str | None:
        """Fetch a secret via the framework provider if bound; else None.

        Never reads os.environ. Trusted deploy-time config source.
        Only an ABSENT secret (MissingSecret) or an unavailable provider is
        treated as None — any other provider error propagates so real
        misconfiguration is not silently swallowed in production.
        """
        try:
            from framework.schemas.invocation_context import current_secrets
        except (ImportError, AttributeError):
            return None
        try:
            from framework import MissingSecret
        except ImportError:  # pragma: no cover - framework always exports it
            MissingSecret = Exception  # type: ignore[assignment,misc]

        provider = current_secrets()
        if provider is None:
            return None
        try:
            return provider.require(key)
        except MissingSecret:
            return None

    def execute(self, state: dict) -> dict:
        parsed = state.get("parsed_request") or {}
        normalized = state.get("normalized_series") or {}
        covariate_matrix = state.get("covariate_matrix") or {}
        horizon = int(parsed.get("horizon", 7))
        mode = parsed.get("inference_mode", "baseline")

        service = self._resolve_service(parsed)
        out = service.forecast(
            normalized_series=normalized,
            covariate_matrix=covariate_matrix,
            horizon=horizon,
            inference_mode=mode,
        )
        emit_trace_event(
            "forecast_computed",
            {"inference_mode": mode, "sku_count": len(out.get("forecast", {}))},
            state,
        )
        return {
            "forecast": out["forecast"],
            "confidence_interval": out["confidence_interval"],
            "status": AgentStatus.SUCCESS.value,
        }
