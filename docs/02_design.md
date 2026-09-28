# Template Design Specification — RET-C2-337

## Position in AgentCore Architecture

- **Agent Class**: RETDemandAnomalyAgent
- **L1 Base**: AgentBaseGraph （Cat 2 = outer `AgentBaseGraph` ＋ `main` スロットに `GraphNode`／inner `domain_workflow_graph.py`）
- **Three-Layer Separation**:
  - State: flat TypedDict composition（Pydantic 禁止 — msgpack 非互換）
  - Node: L1 継承（Template Method: `execute(self, state: dict) -> dict` のみ override）
  - Graph: composition（`register_nodes()` でノード差し替え）

> **設計原則（最重要）:** 異常の「検知」（ステップ1〜4）は**決定論的な Tool Calling**で実行し、**LLM はステップ5のレポート/CSV 生成に限定**する。異常判定そのものを LLM に委ねない（再現性・説明責任・誤判定リスク回避）。

## Architecture Overview

### Cat 2 二層構成

- **Outer graph（`src/graph/graph.py`）:** `RETDemandAnomalyAgent(AgentBaseGraph)`。固定 5 ノード backbone。ドメイン複雑性は `main` スロットの `DemandAnomalyGraphNode`（`GraphNode` サブクラス）に隠蔽。`add_edges()` は override しない。
- **Inner graph（`src/graph/domain_workflow_graph.py`）:** `DemandAnomalyWorkflowGraph(BaseGraph)`。検知ワークフロー（5 ノード）の独自トポロジを実装。`DemandAnomalyGraphNode.get_subgraph()` から呼ばれる。
- **Marketplace adapter（`cli.py`）:** 未検証の runner context を破棄し、環境変数から検証済み scope のみを再構築するとともに、呼び出し単位で資格情報を解決する Azure LLM を `config["llm"]` に注入する。

### Node Configuration（outer backbone）

| Node | Responsibility | Input State | Output State | Inherits/Overrides |
|------|---------------|-------------|--------------|-------------------|
| initialize | schema_version / session_id / trust_level 設定 | user_input | — | InitializeNode (default) |
| pre_process | InputParse: 入力（時系列・covariate・対象 SKU・推論モード）の解析と必須項目検証 | user_input, input_context | validated_input, parsed_request | PreProcessNode（override execute） |
| main | 検知ワークフロー全体を inner graph に委譲 | validated_input, parsed_request | anomaly_result（検知結果＋レポート） | DemandAnomalyGraphNode（GraphNode） |
| post_process | OutputGate: 構造化 JSON / レポート / CSV 整形・スキーマ検証 | anomaly_result（`report_markdown`/`report_csv` を内包） | formatted_output | PostProcessNode（override execute） |
| finalize | response_metadata / total_time_ms 構築 | — | — | FinalizeNode (default) |

### Inner graph（`DemandAnomalyWorkflowGraph` / BaseGraph）ノード構成

| Node | 実行主体 | 主要ロジック | 出力 |
|------|---------|-------------|------|
| data_normalize | Tool（決定論） | 時系列整形・欠損補完・単位統一＋covariate（promo/holiday/price/weather/category）整形 | normalized_series, covariate_matrix |
| covariate_forecast | Tool（TimesFM） | TimesFM 2.5 `forecast_with_covariates`（xreg）を local / BigQuery ML で実行 | forecast, confidence_interval |
| stat_anomaly_detect | **Tool（決定論）** | 実績 vs 予測 CI の統計判定。乖離の符号・大きさで欠品/スパイク/季節性＋深刻度をルール分類 | anomalies[]（anomaly_type / severity） |
| event_attribution | **Tool（ルール）** | 既知イベント KB（プロモ/祝日）と照合し expected（プロモ起因）／真異常を分離 | attributed_anomalies[] |
| report_generate | **LLM** | 構造化検知結果から Markdown レポート / CSV を生成。推奨アクションを言語化・要約 | report_markdown, report_csv |

### Data Flow

```
START → initialize → pre_process(InputParse) → main → {route} → post_process(OutputGate) → finalize → END
                                            ↓ (retry, max 3)
                                          pre_process

main(DemandAnomalyGraphNode) → inner: START → data_normalize → covariate_forecast
        → stat_anomaly_detect → event_attribution → report_generate → END
```

### State Definition

> 共有フィールド（user_input, status, session_id, node_history, error_log, hitl_* 等）は `AgentState` から継承。以下はドメイン追加フィールド。

| Field | Type | Purpose | Required |
|-------|------|---------|----------|
| validated_input | str | pre_process が検証した入力 | Yes |
| parsed_request | dict | 対象 SKU・店舗・期間・推論モード（local/bigquery） | Yes |
| normalized_series | dict | 整形済み時系列 | No |
| covariate_matrix | dict | 整形済み covariate（promo/holiday/price/weather/category） | No |
| forecast | dict | TimesFM 予測値（点予測） | No |
| confidence_interval | dict | TimesFM 予測の信頼区間（lower/upper。`stat_anomaly_detect` が実績との乖離判定に使用） | No |
| anomalies | list | 検知された異常（anomaly_type, severity, affected_skus, confidence_score） | No |
| report_markdown | str | inner: 生成レポート（Markdown） | No |
| report_csv | str | inner: 生成 CSV | No |
| anomaly_result | dict | inner→outer: `DemandAnomalyGraphNode.merge_output()` が帰属済み異常＋`report_markdown`/`report_csv` を 1 つの dict に束ねて outer state へ返す（post_process はこの単一フィールドを読む） | No |
| formatted_output | dict | 最終出力（JSON＋レポート/CSV） | No |

**State Constraints (mandatory):**
- Flat TypedDict only（primitives + JSON-serializable types）
- No JWT, API keys, credentials in State（checkpoint DB leakage）
- InvocationContext via `config["configurable"]` only（not in State）
- No Pydantic models, dataclass, arbitrary Python objects（msgpack incompatible）

## Framework Utilization

### Shared Components Used
- [x] InvocationContext (correlation_id, session_id, permissions, credential handle)
- [x] ConnectionPolicy (retry/timeout strategy) — TimesFM / BigQuery 呼び出し
- [x] SecurityViolationError
- [x] S-2: `_extra_security_gate_input()` — 入力サイズ上限・`max_skus_per_call` 検証（F-3 大量 SKU レイテンシ対策）、対象期間妥当性チェック。`_security_gate_input()` は override しない
- [x] S-3: `_extra_security_gate_output()` — 出力 JSON/CSV の PII 再スキャン（店舗・顧客識別子を含めない）、レポート本文の credential スキャン。`_security_gate_output()` は override しない
- [x] S-4: `emit_trace_event()` — 各 `execute()` 内で異常検知件数・推論モード等のドメインイベントを最低 1 件 emit（`node_start`/`node_complete` は emit しない）

> **S-2/S-3 gate behaviour by node type (ADR-017):**
> - `FunctionNode` subclass（pre/post_process, inner の各ノード）→ framework `@final` gate が自動実行。`_extra_security_gate_*` で拡張のみ
> - `GraphNode`（main = DemandAnomalyGraphNode）→ deliberate no-op（inner ノードの gate が適用済み）

### Composition Pattern

- **Pattern**: GraphNode (subgraph) — outer `main` スロットが inner `DemandAnomalyWorkflowGraph` を包む
- **Composition target**: `src/graph/domain_workflow_graph.py` の `DemandAnomalyWorkflowGraph`
- **Error propagation strategy**: propagate（inner エラーは `SubgraphError` として再送出・fail fast）

#### inner→outer 受け渡し仕様（`get_output()` / `merge_output()`）

inner graph の `DemandAnomalyWorkflowGraph.get_output(state)` と outer `DemandAnomalyGraphNode.merge_output(state, sub_result)` は対で設計する。`anomaly_result` は `attributed_anomalies` ＋ `report_markdown` ＋ `report_csv` を 1 つの dict に束ねた最終ペイロードで、post_process（OutputGate）はこの単一フィールドのみを読む。

```python
# inner: DemandAnomalyWorkflowGraph.get_output(state) -> sub_result
{
  "attributed_anomalies": list,  # [{anomaly_type, severity, affected_skus, confidence_score, attribution}]
  "report_markdown": str,
  "report_csv": str,
  "status": str,                 # AgentStatus 値
}

# outer: DemandAnomalyGraphNode.merge_output(state, sub_result) -> 変更キーのみ
{
  "anomaly_result": {            # 上記 sub_result を束ねた dict
    "attributed_anomalies": sub_result["attributed_anomalies"],
    "report_markdown": sub_result["report_markdown"],
    "report_csv": sub_result["report_csv"],
  },
  "status": sub_result["status"],
}
```

## Security Model（S-1〜S-5）

> 5 層セキュリティの本テンプレでの実配線。S-2/S-3 は `FunctionNode` の `@final` ゲートが自動実行し、ドメイン拡張は `_extra_security_gate_*` フックで行う（`_security_gate_*` 直 override は import 時 TypeError）。

| 層 | 内容 | 本テンプレでの実装箇所 |
|---|---|---|
| **S-1** | トラストゲート（呼び出し元の信頼レベル強制） | 各ノードの `required_trust_level`。最小要件 = `VERIFIED_EXTERNAL`（`config/agent.yaml`）。`BaseNode.__call__` が execute 前に自動強制 |
| **S-2** | 入力ゲート（PII/インジェクション/サイズ） | outer `PreProcessNode._extra_security_gate_input()`：入力サイズ上限＋対象期間妥当性。**`max_skus_per_call` 検証もここ**（既定の `user_input`/`validated_input` PII スキャンに追加） |
| **S-3** | 出力ゲート（credential/PII 除去・スキーマ検証） | outer `PostProcessNode._extra_security_gate_output()`：出力 JSON/CSV の PII 再スキャン（店舗・顧客識別子除去）＋レポート本文の credential スキャン。`s3_gate_enabled: true` |
| **S-4** | 監査トレース | 各 `execute()` 内で `emit_trace_event()`（異常検知件数・推論モード等のドメインイベントを最低 1 件。`node_start`/`node_complete` は自動のため emit しない） |
| **S-5** | **レート制限 / リソース上限**（大量 SKU・推論コスト暴走の抑止） | **`max_skus_per_call`（config 既定値）で 1 リクエストあたりの対象 SKU 数を上限化**し、S-2 入力ゲート（`PreProcessNode._extra_security_gate_input()`）で超過を拒否。超過分は呼び出し元でページネーション（バッチ分割）する前提。TimesFM 推論コスト・レイテンシ（F-3）の構造的上限として機能する |

> **S-5 ディスクレーマー:** 本テンプレの S-5 は「1 invoke あたりの SKU 数上限（`max_skus_per_call`）」によるアプリ層のレート制限であり、プラットフォーム全体の API レート制限（ゲートウェイ層）を代替しない。56,000 店舗 × 全 SKU の一括監視は呼び出し元でのバッチ分割を必須とし、本エージェント単体での無制限実行は想定しない。

## TimesFM 推論モード（計算は AG 外部）

> **アーキテクチャ原則**: TimesFM 推論（GPU・モデル重み）は **AG プロセスの外部リソース**。
> AG は軽量に保ち（整形・決定論検知・レポート）、推論は外部を**呼ぶだけ**。

- `parsed_request["inference_mode"]` でトグル:
  - **`remote`（本番推奨）**: 外部 TimesFM サービング endpoint へ HTTP POST（`RemoteTimesFMCaller`）。AG は計算を持たない。**🔒 endpoint URL・認証は信頼できる secret（デプロイ時）からのみ取得し、リクエスト入力からは取らない＋https 必須**（SSRF・トークン漏洩防止）。
  - **`bigquery`（本番・GCP）**: BigQuery ML（外部）。
  - `baseline`（既定）: 依存ゼロの決定論 in-process（CI/デモ）。
  - `local`: in-process 実 TimesFM = **開発検証専用**（本番非推奨。計算外部原則に反する）。
- `covariate_forecast` ノードは `src/services/` のサービス層経由で呼び出し、ノードにエンドポイント・認証情報を直書きしない（secrets は `current_secrets().require()`、`os.environ` 禁止）。
- `RemoteTimesFMCaller` は `http_post` 注入で**実ネットワークなしに CI テスト可**（本番経路が CI で検証される）。
- **F-1（covariate 誤発火精度）/ F-2（顧客プロモカレンダー可用性）は実装・テストフェーズの検証項目**（着手前提条件ではない）。プロモ/非プロモ分解の誤検知率は integration テストで測定し、運用前に利用環境のデータでも評価する

## Import Isolation Confirmation
- [x] Template does not import agenticstar-platform SDK (Level 0)
- [x] Import targets: framework/ and shared/ only（no agents/base/ required）

## Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| L1 base type | AgentBaseGraph | AutonomousBaseGraph | **AgentBaseGraph** | 自律ループ不要の固定多段ワークフロー（Cat 2）。think→act ループは過剰 |
| Composition pattern | Standalone | GraphNode (subgraph) | **GraphNode** | 検知ワークフローが 5 ノードと多段のため inner graph に隠蔽し outer backbone を簡潔に保つ |
| 異常判定の主体 | LLM 分類 | 決定論 Tool（統計判定） | **決定論 Tool** | 再現性・説明責任・誤判定リスク回避（論点A）。LLM はレポート生成のみ |
| 外生要因の扱い | 後処理で除外 | `forecast_with_covariates`（xreg）で予測に織込み | **xreg 織込み** | プロモ/祝日/価格/天候を予測に内包し誤発火を構造的に抑制（論点B） |
