# RET-C2-337 — Retail Demand Anomaly Detection Agent

> **Category**: Cat 2 (domain workflow)
> **Industry**: RET

## Overview

Retail Demand Anomaly Detection Agent compares recent per-SKU demand against a forecast and its
confidence interval, then reports the deviations a planner should actually look at. It normalises
the submitted series, produces a forecast with covariates (promotion, holiday, price discount and
similar signals), flags points outside the interval, decides whether an event already explains the
deviation, and renders the result as structured JSON plus a Markdown report and a CSV.

It suits the daily or weekly pass over sales data: a demand planner picking out which spikes are
worth investigating, an inventory planner extracting stockout-risk candidates, or an analyst who
wants the flagged SKUs in a spreadsheet with the promotion signals already accounted for.

Detection is entirely deterministic and auditable, and the design keeps it that way:

- **The model never decides.** Normalisation, forecasting, interval comparison and event
  attribution are all rule-based. The language model is used at one step only — narrating an
  already-computed result into Markdown. It cannot introduce, remove or re-grade an anomaly.
- **The CSV stays deterministic** even when the narrative does not, so the machine-readable output
  is unaffected by the model.
- **A degraded narration is visible, not silent.** With no model injected, or when the call fails
  or returns nothing, a deterministic renderer takes over; `narration_mode` says which path ran and
  `degradation_reason` appears when a wired model did not deliver.
- **Planned variation is separated from real anomalies.** A demand-up deviation that coincides
  with an active promotion, holiday or discount signal is marked `expected` rather than
  `true_anomaly`; only true anomalies are counted in the summary. Demand-*down* deviations are
  never suppressed this way — a promotion does not explain demand falling.

The agent classifies deviations. It does not validate causal explanations, and it does not make
replenishment, pricing or other business decisions.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Integration

Provide:

- an AgentCore runtime;
- a forecasting backend, selected per request by `inference_mode`:

  | Mode | Backend |
  |---|---|
  | `baseline` (default) | In-process deterministic forecast. No dependency, no external call — the safe default for CI and demos, and the fallback shape |
  | `remote` | An external TimesFM serving endpoint over HTTPS. Requires `timesfm_endpoint` and `timesfm_api_token` from the trusted secret provider — never from the request. A non-HTTPS or host-less endpoint is rejected |
  | `bigquery` | External BigQuery ML forecasting. Requires `google-cloud-bigquery`, imported lazily |
  | `local` | The real model in-process. Development and offline verification only — needs the `local` extra, and is not the production shape |

- optionally, a `BaseLLM` implementation through `config["llm"]` for the narration step, with an
  optional `system_prompt` alongside it. Both are forwarded from the outer graph into the inner
  workflow. Without a model the deterministic renderer runs;
- caller authentication in front of the entry adapter.

**Where the parameters actually come from is worth being precise about.** `config/config.yaml`
holds `inference_mode`, `max_skus_per_call`, `horizon`, `max_retry` and `timeout_seconds`, but the
standalone server in `src/api/server.py` constructs the graph with **no configuration**, and the
nodes read `inference_mode` and `horizon` from the *request body* (defaulting to `baseline` and
`7`), while the SKU ceiling is a module constant of 500 in `src/nodes/pre_process_node.py`. Treat
`config.yaml` as the documented defaults to wire up for your deployment — read it and pass it to
the graph, or change the constant — rather than as values the shipped standalone path already
honours.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent is built for the platform and connects to it at start-up. Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Supplied by the platform's own package registry at build time. It is not published on public PyPI, so `pip install -e .` alone does not fetch it. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The agent does not fall back to a degraded mode when the framework is missing: `src/api/server.py`
imports `framework.*` and `shared.*` at module scope and builds, compiles and provisions the graph
at import time, so without the AgentCore package the process fails to start. With the framework
installed but no platform around it the server does start and answers `/health`, and a `baseline`
request completes end-to-end with no external dependency — but it configures no caller
authentication, so `/invoke` answers **401** until `INVOKE_AUTH_TOKEN` is set or verifying
middleware is placed in front, and it injects no model, so the report comes back with
`narration_mode: "deterministic"`. The test suite runs without a platform connection.

## Input

`POST /invoke` takes a JSON body with `input` — the analysis request **as a JSON string** — and an
optional `session_id`.

Callers must be authenticated. The entry adapter uses a trust level stamped by upstream middleware
if one is present; otherwise, when `INVOKE_AUTH_TOKEN` is set on the server environment, a matching
`Authorization: Bearer` header promotes the caller to `VERIFIED_EXTERNAL` (a mismatched token is
**401**). A caller that is neither is rejected with **401**.

```bash
curl -X POST http://127.0.0.1:8000/invoke \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${INVOKE_AUTH_TOKEN}" \
  -d '{
    "input": "{\"series\":{\"SKU1\":[10,11,9,10,12,10,11,10,11,9,10,12,10,40]},\"covariates\":{\"promo\":[0,0,0,0,0,0,0,0,0,0,0,0,0,0]},\"target_skus\":[\"SKU1\"],\"period\":\"2026-06\",\"inference_mode\":\"baseline\",\"horizon\":7}",
    "session_id": "demand-review-2026-06"
  }'
```

The request object carries:

| Field | Meaning |
|---|---|
| `series` | Required, non-empty. `{sku: [values…]}` — the observed demand history |
| `covariates` | `{signal: [values…]}`. `promo`, `holiday` and `price_discount` are the signals attribution reads; the last value being greater than zero means the event is active |
| `target_skus` | Which SKUs to analyse. Defaults to every key in `series` |
| `period` | Label carried into the report |
| `inference_mode` | One of the four backends above. Defaults to `baseline` |
| `horizon` | Forecast points. Defaults to `7` |

Input fails closed: malformed JSON, or a `series` that is absent or not a non-empty object, ends
as `status=error` with the reason in the error log. A request naming more SKUs than the ceiling is
refused before analysis, with the instruction to paginate upstream. `inference_mode: "remote"` with
no trusted endpoint secret is an error, never a silent fall back to `baseline`.

## Output

The standard AgentCore envelope is returned. `output` carries:

| Field | Contents |
|---|---|
| `anomaly_detected` / `anomaly_count` | Whether any *true* anomaly was found, and how many. Deviations attributed to a promotion are excluded from both |
| `anomalies` | One entry per true anomaly: `affected_skus`, `anomaly_type`, `severity`, `recommended_action`, `confidence_score` |
| `report_markdown` | The narrative report — model-written or deterministic |
| `report_csv` | Always deterministic: sku, anomaly type, severity, deviation, confidence, attribution, recommended action |
| `narration_mode` | Which narration path ran |
| `degradation_reason` | Present only when a wired model did not deliver. Absent means nothing degraded |

Three anomaly types are produced, by rule: `demand_spike` (above the interval, deviation ratio at
or above 0.5), `stockout_risk` (above the interval, smaller deviation), and `seasonal_anomaly`
(below the interval). A point inside the confidence interval is not an anomaly. `confidence_score`
is the absolute deviation ratio capped at 1.0 — it measures how far outside the band the point
sits, not how likely the finding is to be correct.

Values are illustrative; real output depends on the series supplied.

## Security and Limitations

The template's own nodes declare their trust level explicitly rather than inheriting a permissive
default; the narration node, which issues a billable external call, requires verified-external
trust. Forecast-backend credentials come only from the trusted secret provider — a request cannot
supply an endpoint — and a remote endpoint must be an HTTPS URL with a host.

Known limits, recorded rather than open:

- **Event attribution is a deliberately conservative heuristic.** "The last covariate value is
  greater than zero" is what makes a promotion count as active. A promotion that ended mid-window,
  or one absent from the covariates, will not suppress the anomaly it explains — and a stale
  positive signal will suppress one it does not.
- **`baseline` is a stand-in, not a forecast.** The default mode is a deterministic in-process
  computation for CI and demonstration. Production runs `remote` or `bigquery`; `local` exists for
  offline development and is not the production shape.
- **The agent classifies deviations, it does not explain them.** `recommended_action` is a fixed
  string attached to each anomaly type, not advice derived from the data.
- **Input is a SKU-level series.** The template assumes aggregated, non-identifying structured
  data; it is not built for records that identify individuals.
- **One call, bounded work.** Requests above the SKU ceiling are rejected rather than truncated —
  paginate upstream.

Output is machine-generated and assistive. It does not substitute for expert review, and any
replenishment or pricing decision should be made by a person against the primary data.

## Release

- Version: 1.0.0
- Category: fixed multi-step domain workflow (Cat 2) — an outer pipeline whose main step wraps an
  inner detection workflow
- Generation mode: `llm` — the report narration only; detection, attribution and the CSV are
  deterministic
- HITL: not enabled
- Persistence: none — no checkpointer is configured
- Secrets: none declared in the manifest. `remote` mode resolves `timesfm_endpoint` and
  `timesfm_api_token` through the trusted secret provider at request time; the entry adapter
  additionally reads `INVOKE_AUTH_TOKEN` from the environment when caller-token authentication is
  in use
- External API clients: deployment-injected; none bundled

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection, without a forecasting endpoint and without a language
model, using `baseline` mode and fakes. A local `uvicorn src.api.server:app` run behaves the same
way once caller authentication is configured.

## Documentation

`docs/02_design.md` carries the node flow, the state schema, the detection rules and the
security-layer mapping. `docs/03_test_spec.md` records the test specification and results,
including the narration seam.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent configuration
docs/         design and test specifications
```

See `docs/02_design.md` for the design and `docs/03_test_spec.md` for the test specification.

## Customising

1. Adjust `config/` for your own environment and policies.
2. Replace the knowledge sources and sample data with your own.
3. Review the node implementations under `src/nodes/` for domain-specific logic.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
