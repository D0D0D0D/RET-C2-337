"""AgentCore Platform v1.0 — LLM seam for report narration.

`report_generate` is the ONLY LLM step in this template (docs/02_design.md §3:
detection is deterministic Tool calling; the LLM narrates the result). A
`BaseLLM` instance injected through the graph config (`config["llm"]`, forwarded
outer graph -> GraphNode -> inner workflow graph) writes the Markdown report from
the already-computed, auditable detection result.

When no LLM is injected the module falls back to a deterministic Markdown
renderer so the template still runs in CI, unit tests and un-provisioned
environments. The fallback is NOT silent: `render_report()` reports the
narration mode back to the caller, which surfaces it in the agent output, so a
deterministic report can never be mistaken for an LLM-written one.

Guardrails:
  - Detection logic must NEVER depend on this module — it only narrates an
    already computed, auditable result (ADR / Issue A guardrail).
  - The authoritative machine-readable record (`anomalies` in formatted_output,
    and `report_csv`) stays deterministic. The LLM writes prose only, so a
    narration failure can never change what was detected.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

# A renderer is: (anomalies: list[dict], meta: dict) -> markdown: str
Renderer = Callable[[list, dict], str]

# Narration modes reported alongside the rendered Markdown.
NARRATION_LLM = "llm"
NARRATION_DETERMINISTIC = "deterministic"
NARRATION_CUSTOM = "custom"

# Degradation reasons. Empty string == no degradation.
#   Running without an injected LLM is NOT a degradation: it is the documented
#   CI / un-provisioned behaviour, reported through the narration mode instead.
REASON_NARRATION_EMPTY = "llm_narration_empty"
REASON_NARRATION_FAILED = "llm_narration_failed"
REASON_NARRATION_INCOMPLETE = "llm_narration_incomplete"

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def deterministic_markdown(anomalies: list[dict], meta: dict[str, Any]) -> str:
    """Deterministic Markdown narration of the detection result."""
    period = meta.get("period", "N/A")
    mode = meta.get("inference_mode", "baseline")
    true_anomalies = [a for a in anomalies if a.get("attribution") == "true_anomaly"]
    expected = [a for a in anomalies if a.get("attribution") == "expected"]

    lines = [
        "# 需要異常検知レポート",
        "",
        f"- 対象期間: {period}",
        f"- 推論モード: {mode}",
        f"- 検知異常数: {len(true_anomalies)}（プロモ起因として除外: {len(expected)}）",
        "",
        "## 検知された異常（要対応）",
        "",
    ]
    if not true_anomalies:
        lines.append("対応が必要な異常は検知されませんでした。")
    else:
        lines.append("| SKU | 種別 | 深刻度 | 乖離 | 信頼度 | 推奨アクション |")
        lines.append("|---|---|---|---|---|---|")
        for a in sorted(true_anomalies, key=lambda x: _SEVERITY_ORDER.get(x.get("severity", "low"), 9)):
            lines.append(
                f"| {a.get('sku','')} | {a.get('anomaly_type','')} | {a.get('severity','')} "
                f"| {a.get('deviation','')} | {a.get('confidence_score','')} "
                f"| {a.get('recommended_action','')} |"
            )
    return "\n".join(lines)


def anomalies_to_csv(anomalies: list[dict]) -> str:
    """Deterministic CSV of the attributed anomalies (no PII)."""
    header = "sku,anomaly_type,severity,deviation,confidence_score,attribution,recommended_action"
    rows = [header]
    for a in anomalies:
        rows.append(
            ",".join(
                str(a.get(k, ""))
                for k in (
                    "sku",
                    "anomaly_type",
                    "severity",
                    "deviation",
                    "confidence_score",
                    "attribution",
                    "recommended_action",
                )
            )
        )
    return "\n".join(rows)


REPORT_SYSTEM_PROMPT = (
    "あなたは小売の需要分析レポートを書くアナリストです。"
    "与えられた検知結果 JSON だけを根拠に、日本語の Markdown レポート本文のみを返してください。\n"
    "厳守事項:\n"
    "- 検知結果に無い SKU・数値・原因を作り出さない。異常判定をやり直さない。\n"
    '- attribution が "expected" の項目はプロモ起因として除外し、要対応として挙げない。\n'
    '- attribution が "true_anomaly" の項目は SKU・種別・深刻度に触れ、'
    "recommended_action を現場が実行できる言葉に言い換える。\n"
    "- 見出しと表または箇条書きを使う。前置き・後書き・コードフェンスは付けない。"
)


def build_report_messages(anomalies: list[dict], meta: dict[str, Any]) -> list[dict]:
    """Build the narration prompt. Lives in src/ — never in tests."""
    payload = {
        "meta": {
            "period": meta.get("period", ""),
            "inference_mode": meta.get("inference_mode", "baseline"),
        },
        "anomalies": anomalies,
    }
    user = (
        "以下は決定論的な統計検知が確定させた結果です。この内容だけを使ってレポート本文を書いてください。\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": REPORT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def missing_true_anomaly_skus(markdown: str, anomalies: list[dict]) -> list[str]:
    """SKUs the detector confirmed as true anomalies but the narration omitted.

    Grounding guard: the LLM writes prose, so it can summarise an anomaly away.
    An ops report that drops a detected anomaly is worse than the deterministic
    table, so an incomplete narration is refused rather than published. This
    checks omission only — it cannot prove the absence of invented content, which
    is why the authoritative record (formatted_output["anomalies"], report_csv)
    stays deterministic and is never taken from the LLM.
    """
    required = []
    for anomaly in anomalies:
        sku = str(anomaly.get("sku", ""))
        if sku and anomaly.get("attribution") == "true_anomaly":
            required.append(sku)
    missing = []
    for sku in required:
        # ASCII token boundaries, not \b: a Japanese character is a word char to
        # re, so \b would refuse to match "SKU2の需要" and degrade every Japanese
        # report. Bounding on [0-9A-Za-z_] also stops "SKU2" matching "SKU20"
        # and stops a 1-char SKU being satisfied by an unrelated letter.
        if not re.search(rf"(?<![0-9A-Za-z_]){re.escape(sku)}(?![0-9A-Za-z_])", markdown):
            missing.append(sku)
    return missing


def extract_content(response: Any) -> str:
    """Read the assistant text out of a BaseLLM.complete() return value."""
    if isinstance(response, dict):
        return str(response.get("content") or "").strip()
    return str(response or "").strip()


class LLMReportClient:
    """Report narration client.

    Injection order:
      1. `renderer=` — a caller-supplied callable (extension point, mode "custom")
      2. `llm=` — a `BaseLLM`; the module builds the prompt and reads the reply
      3. neither — deterministic Markdown (CI / un-provisioned deployments)
    """

    def __init__(
        self,
        renderer: Renderer | None = None,
        llm: Any | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._renderer = renderer
        self._llm = llm
        self._system_prompt = system_prompt

    @property
    def uses_llm(self) -> bool:
        return self._renderer is None and self._llm is not None

    def render_report(self, anomalies: list[dict], meta: dict[str, Any]) -> tuple[str, str, str]:
        """Render the Markdown body.

        Returns `(markdown, narration_mode, degradation_reason)`. No state is
        kept on the instance — node objects are shared across invocations
        so the mode travels with the return value.
        """
        if self._renderer is not None:
            return self._renderer(anomalies, meta), NARRATION_CUSTOM, ""
        if self._llm is None:
            return deterministic_markdown(anomalies, meta), NARRATION_DETERMINISTIC, ""

        messages = build_report_messages(anomalies, meta)
        if self._system_prompt is not None:
            messages[0] = {"role": "system", "content": self._system_prompt}
        try:
            content = extract_content(self._llm.complete(messages))
        except Exception as exc:  # noqa: BLE001 — narration must not fail detection
            # Only the exception TYPE is propagated: a provider message can carry
            # request echoes / credentials and must never reach State.
            return (
                deterministic_markdown(anomalies, meta),
                NARRATION_DETERMINISTIC,
                f"{REASON_NARRATION_FAILED}:{type(exc).__name__}",
            )
        if not content:
            return (
                deterministic_markdown(anomalies, meta),
                NARRATION_DETERMINISTIC,
                REASON_NARRATION_EMPTY,
            )
        if missing_true_anomaly_skus(content, anomalies):
            return (
                deterministic_markdown(anomalies, meta),
                NARRATION_DETERMINISTIC,
                REASON_NARRATION_INCOMPLETE,
            )
        return content, NARRATION_LLM, ""

    def render_markdown(self, anomalies: list[dict], meta: dict[str, Any]) -> str:
        """Markdown body only (mode/reason dropped)."""
        return self.render_report(anomalies, meta)[0]

    def render_csv(self, anomalies: list[dict]) -> str:
        """Always deterministic — the machine-readable record is never narrated."""
        return anomalies_to_csv(anomalies)
