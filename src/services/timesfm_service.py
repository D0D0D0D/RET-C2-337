"""AgentCore Platform v1.0 — TimesFM forecasting service.

Service layer: wraps TimesFM 2.5 zero-shot forecasting with exogenous
covariates. Nodes call this; this hides the inference backend behind a single
interface so domain nodes never touch endpoints or credentials.

ARCHITECTURE (important): TimesFM compute is an EXTERNAL resource — the agent
does NOT host the model. The agent stays light (input shaping + deterministic
detection + report) and CALLS an external inference resource. The production
path is therefore `remote` (HTTP serving endpoint) or `bigquery` (BigQuery ML),
where the GPU/model weights live OUTSIDE the agent process.

Backend selection is driven by `inference_mode`:
  - "remote"   -> EXTERNAL TimesFM serving endpoint over HTTP (RemoteTimesFMCaller).
                  PRODUCTION path. Agent holds no model/compute; URL+auth via
                  config/secrets. Fully testable with an injected http_post.
  - "bigquery" -> EXTERNAL BigQuery ML (`AI.FORECAST`/`ML.FORECAST`). Production
                  path for GCP-native customers. Injection point (needs GCP).
  - "baseline" -> deterministic seasonal-naive + covariate-linear forecaster.
                  DEFAULT; dependency-free, reproducible, in-process. For CI/tests,
                  demos, and graceful fallback. NOT the real model.
  - "local"    -> in-process real TimesFM (timesfm[torch]). DEV/OFFLINE ONLY —
                  this loads the 200M model INTO the agent process, which violates
                  the "compute is external" principle. Use only for local
                  verification; never the production deployment shape.

The agent should run `remote`/`bigquery` in production. `baseline` is the safe
in-process default; `local` is a developer convenience for offline validation.
"""

from __future__ import annotations

from typing import Any, Callable

# A backend is a callable: (series, covariate_matrix, horizon) -> (point, ci)
Backend = Callable[[list[float], dict[str, list[float]], int], "tuple[list[float], dict[str, list[float]]]"]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _stdev(xs: list[float], mu: float) -> float:
    if len(xs) < 2:
        return 0.0
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return var ** 0.5


def baseline_forecast(
    series: list[float],
    covariate_matrix: dict[str, list[float]],
    horizon: int,
) -> tuple[list[float], dict[str, list[float]]]:
    """Deterministic seasonal-naive + covariate-linear forecaster.

    - Level: trailing mean of the series.
    - Seasonality: weekly (period 7) additive offset if enough history.
    - Covariates: each numeric covariate's most recent value nudges the point
      forecast by a small linear weight (promo/price/weather effect proxy),
      so that `forecast_with_covariates` semantics (exogenous adjustment) are
      preserved deterministically.
    - Confidence interval: point ± 2 * trailing stdev (95% proxy band).

    Reproducible (no randomness) — this is what makes detection auditable.
    """
    if not series:
        return [0.0] * horizon, {"lower": [0.0] * horizon, "upper": [0.0] * horizon}

    level = _mean(series[-min(len(series), 14):])
    sigma = _stdev(series, _mean(series))
    period = 7
    season = [0.0] * period
    if len(series) >= period:
        base = _mean(series)
        for i in range(period):
            same_phase = series[i::period]
            if same_phase:
                season[i] = _mean(same_phase) - base

    # covariate linear adjustment (last observed value of each covariate)
    cov_adj = 0.0
    for name, values in (covariate_matrix or {}).items():
        if not values:
            continue
        try:
            last = float(values[-1])
        except (TypeError, ValueError):
            continue
        # promo/price lift covariates push demand up; default small weight 0.05
        weight = 0.10 if name in ("promo", "price_discount") else 0.05
        cov_adj += weight * last

    point = []
    lower = []
    upper = []
    n = len(series)
    for h in range(horizon):
        phase = (n + h) % period
        p = level + season[phase] + cov_adj
        point.append(round(p, 4))
        lower.append(round(p - 2.0 * sigma, 4))
        upper.append(round(p + 2.0 * sigma, 4))
    return point, {"lower": lower, "upper": upper}


# Module-level cache: loading + compiling the 200M model is expensive (~60s
# cold, weights downloaded from HuggingFace), so do it once per process.
_TIMESFM_MODEL = None
# quantile-head column indices used for the confidence band (~10% / ~90%).
_Q_LOWER_COL = 1
_Q_UPPER_COL = 8


def _load_timesfm_model():
    """Lazily load + compile the real TimesFM 2.5 200M (torch) model.

    Verified to run on CPU (forecast + forecast_with_covariates). The xreg
    covariate path additionally needs `jax` (CPU is sufficient) + scikit-learn;
    `return_backcast=True` is required by XReg. Requires the `local` extra
    (`pip install ".[local]"`). Never imported at module top level so the
    template imports cleanly without these optional deps (CI runs `baseline`).
    """
    global _TIMESFM_MODEL
    if _TIMESFM_MODEL is not None:
        return _TIMESFM_MODEL
    try:
        import timesfm
    except ImportError as exc:  # pragma: no cover - optional heavy dep
        raise RuntimeError(
            "inference_mode='local' requires the 'local' extra "
            "(pip install \".[local]\": timesfm[torch] + jax + scikit-learn). "
            "Use inference_mode='baseline' where these are unavailable."
        ) from exc
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        timesfm.TimesFM_2p5_200M_torch.DEFAULT_REPO_ID
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=512,
            max_horizon=64,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            return_backcast=True,  # required by XReg covariate path
        )
    )
    _TIMESFM_MODEL = model
    return model


def _shape_covariate(values: list[float], context_len: int, horizon: int) -> list[float] | None:
    """Return a covariate series of length context+horizon, or None if unusable.

    Dynamic covariates must cover both history and the forecast window. If the
    caller supplied future values (len == context+horizon) use them as-is; if
    only history (len == context) carry the last value forward over the horizon.
    """
    need = context_len + horizon
    vals = [float(v) for v in values]
    if len(vals) == need:
        return vals
    if len(vals) == context_len and vals:
        return vals + [vals[-1]] * horizon
    return None


def _local_backend(series, covariate_matrix, horizon):
    """DEV/OFFLINE-ONLY in-process TimesFM 2.5 backend (torch, CPU-capable).

    ⚠️ This loads the 200M model INTO the agent process — it violates the
    "compute is external" principle and is NOT the production deployment shape.
    Use only for local verification that the real model runs; production uses
    `remote` (RemoteTimesFMCaller) or `bigquery`. Kept because it lets us prove
    the real TimesFM path works end-to-end offline.

    Uses forecast_with_covariates when covariates can be shaped for the full
    timeline (xreg + timesfm), else plain forecast(). Returns (point, ci) with
    the confidence band taken from the model's quantile head.
    """
    import numpy as np

    model = _load_timesfm_model()
    ctx = len(series)

    dynamic_cov = {}
    for name, values in (covariate_matrix or {}).items():
        shaped = _shape_covariate(values, ctx, horizon)
        if shaped is not None:
            dynamic_cov[name] = [shaped]

    if dynamic_cov:
        points, quantiles = model.forecast_with_covariates(
            inputs=[list(series)],
            dynamic_numerical_covariates=dynamic_cov,
            xreg_mode="xreg + timesfm",
            ridge=0.0,
            force_on_cpu=True,
        )
        point = np.asarray(points[0]).ravel()[:horizon]
        q = np.asarray(quantiles[0])
    else:
        pt, qt = model.forecast(horizon=horizon, inputs=[np.asarray(series, dtype=float)])
        point = np.asarray(pt[0]).ravel()[-horizon:]
        q = np.asarray(qt[0])[-horizon:]

    lower = q[:, _Q_LOWER_COL].ravel()[:horizon]
    upper = q[:, _Q_UPPER_COL].ravel()[:horizon]
    return (
        [round(float(x), 4) for x in point],
        {
            "lower": [round(float(x), 4) for x in lower],
            "upper": [round(float(x), 4) for x in upper],
        },
    )


def _bigquery_backend(series, covariate_matrix, horizon):
    """BigQuery ML backend (injection point).

    Lazily imports google-cloud-bigquery and runs ML.FORECAST against the
    customer's trained TimesFM model. Credentials come from the framework
    secrets provider (ctx.secrets.require) — never os.environ.
    """
    try:
        from google.cloud import bigquery  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on optional heavy dep
        raise RuntimeError(
            "inference_mode='bigquery' requires google-cloud-bigquery."
        ) from exc
    # TODO(impl): run `SELECT * FROM ML.FORECAST(MODEL ..., STRUCT(horizon ...))`
    raise NotImplementedError("BigQuery ML backend not yet wired — see docs/07")


_BACKENDS: dict[str, Backend] = {
    "baseline": baseline_forecast,
    "local": _local_backend,
    "bigquery": _bigquery_backend,
}


class RemoteTimesFMCaller:
    """PRODUCTION backend: call an EXTERNAL TimesFM serving endpoint over HTTP.

    The agent holds NO model and NO compute — it POSTs the (single-series)
    context + covariates to an external inference service and parses the
    forecast back. This is the architecturally-correct shape: GPU/model weights
    live OUTSIDE the agent process, and the agent stays light.

    Wire-in: the node builds this with the endpoint URL (from config/request)
    and an auth token (from the framework secrets provider). `http_post` is
    injectable so the production path is fully unit-testable in CI without any
    real network call.

    Expected endpoint contract (JSON):
        request : {"series": [floats], "covariates": {name: [floats]}, "horizon": int}
        response: {"forecast": [floats(horizon)],
                   "confidence_interval": {"lower": [floats], "upper": [floats]}}
    """

    def __init__(
        self,
        endpoint: str,
        auth_token: str | None = None,
        http_post: Callable[..., dict] | None = None,
        timeout: int = 30,
    ) -> None:
        if not endpoint:
            raise ValueError("RemoteTimesFMCaller requires a non-empty endpoint URL")
        self._endpoint = endpoint
        self._auth_token = auth_token
        self._http_post = http_post or self._default_post
        self._timeout = timeout

    @staticmethod
    def _default_post(url: str, payload: dict, headers: dict, timeout: int) -> dict:
        import json
        import urllib.request

        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec - URL from config
            return json.loads(resp.read())

    def __call__(self, series, covariate_matrix, horizon):
        headers = {"Content-Type": "application/json"}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        payload = {
            "series": [float(x) for x in series],
            "covariates": {k: [float(v) for v in vs] for k, vs in (covariate_matrix or {}).items()},
            "horizon": int(horizon),
        }
        resp = self._http_post(self._endpoint, payload, headers, self._timeout)
        point = [float(x) for x in resp["forecast"]]
        ci = resp.get("confidence_interval") or {}
        lower = [float(x) for x in ci.get("lower", point)]
        upper = [float(x) for x in ci.get("upper", point)]
        return point, {"lower": lower, "upper": upper}


class TimesFMService:
    """Dual-mode TimesFM forecasting service.

    Inject a custom backend for testing or to wire a real model:
        TimesFMService(backend=my_callable)
    Otherwise the backend is resolved from `inference_mode` per call, defaulting
    to the deterministic `baseline` backend.
    """

    def __init__(self, backend: Backend | None = None) -> None:
        self._backend = backend

    def forecast(
        self,
        normalized_series: dict[str, list[float]],
        covariate_matrix: dict[str, list[float]],
        horizon: int,
        inference_mode: str = "baseline",
    ) -> dict[str, Any]:
        """Return {forecast: {sku: [..]}, confidence_interval: {sku: {lower,upper}}}."""
        backend = self._backend or _BACKENDS.get(inference_mode, baseline_forecast)
        forecast: dict[str, list[float]] = {}
        ci: dict[str, dict[str, list[float]]] = {}
        for sku, series in normalized_series.items():
            point, band = backend(series, covariate_matrix, horizon)
            forecast[sku] = point
            ci[sku] = band
        return {"forecast": forecast, "confidence_interval": ci}
