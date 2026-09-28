# Test Specification — RET-C2-337

## テスト方針

- **検知ロジックは決定論**のため、合成データで期待値を厳密に検証できる（再現性＝moat）。
- LLM/TimesFM の外部依存はサービス層の注入シーム（`TimesFMService` / `LLMReportClient`）で
  デフォルト決定論バックエンドに差し替え、CI は外部依存なしで緑にする。
- セキュリティ（S-2/S-3/S-5）と境界（PB: import isolation / state safety）を必ず検証する。

## テストレイヤ

| レイヤ | ファイル | 対象 |
|---|---|---|
| unit | `tests/unit/test_nodes.py` | ノード契約（execute シグネチャ）・data_normalize・stat_anomaly_detect・event_attribution・S-5 ゲート |
| unit | `tests/unit/test_services.py` | baseline_forecast の決定性・covariate リフト・TimesFMService 注入・LLMReportClient |
| integration | `tests/integration/test_end_to_end.py` | フル invoke（happy path / backbone / S-5 / 不正入力）＋ **F-1 誤検知率測定** |
| proof_of_boundary | `tests/proof_of_boundary/*` | PB-4 import isolation・PB-2/5 state safety |

## 主要テストケース

| TC | 内容 | 期待 |
|---|---|---|
| TC-01 | 非プロモのスパイク（SKU2=40） | `demand_spike`・severity=high を 1 件検知 |
| TC-02 | CI 内変動 | 異常検知ゼロ |
| TC-03 | プロモ起因スパイク | `expected` に帰属＝true_anomaly に数えない |
| TC-04 | S-5: 600 SKU | status=ERROR（`max_skus_per_call` 超過拒否） |
| TC-05 | 不正 JSON / 空 series | status=ERROR |
| TC-06 | レポート/CSV 生成 | Markdown 表＋CSV ヘッダを出力、PII なし |
| TC-08 | `required_trust_level` 宣言・強制 | 不足 trust は S-1 で拒否し、PB-6 の `test_s1_denial_refuses_execution_before_execute` が `execute()` が呼ばれなかったことを証明する |

## F-1 誤検知率測定（条件付きGo 検証項目）

`TestF1FalsePositiveSuppression` で合成データにより 2 指標を測定・assert する：

- **誤発火抑制**: 20 SKU の「プロモ起因スパイク」に対し、true_anomaly として誤флаグされる
  False Positive 率 **≤ 10%**（`test_promo_spikes_are_suppressed`）。
- **再現率（recall）**: 同じスパイクをプロモなしで与えたとき、真の異常として **≥ 90%** 検知
  （`test_true_spikes_still_detected_without_promo`）。

> 本測定は合成データによる構造検証。実際の POS／プロモカレンダーにおける精度（F-1/F-2）は
> 本テストの対象外であり、導入環境のデータを用いた運用前評価が必要となる（docs/02 §TimesFM 推論モード参照）。

## 実 LLM 1 回転検証（generation_mode: "llm" の実測）

`config/agent.yaml` は `generation_mode: "llm"` を宣言している。宣言が実装と一致することを
実 LLM で 1 回転させて確認した記録。

| 項目 | 内容 |
|---|---|
| 実施日 | 2026-08-20 |
| SDK | `agenticstar-agentcore==1.0.1` |
| 使用 LLM | ローカル CLI 経由の Azure OpenAI（`BaseLLM` 実装を検証用に注入） |
| 注入シーム | `config["llm"]`（本番と同一。外側 `RETDemandAnomalyAgent.register_nodes()` → `DemandAnomalyGraphNode._parent_config()` → 内側 `DemandAnomalyWorkflowGraph.register_nodes()` → `ReportGenerateNode`） |
| LLM 呼び出し回数 | **1**（`report_generate` のみ。検知 4 ノードは決定論のまま） |
| 結果 | ✅ `status=success` / `narration_mode="llm"` / `degradation_reason` なし |
| 非空だった出力キー | `anomaly_detected`, `anomaly_count`, `anomalies`, `report_markdown`, `report_csv`, `narration_mode` |
| 生成物 | `report_markdown` は LLM が書いた日本語レポート（検知した SKU2 の needs-action 行を含む）。`report_csv` と `anomalies` は決定論のまま |

### 検証前に見つかった欠陥（本 MR で修正）

実測の結果、**宣言と実装が一致していなかった**ことが判明した：

- `src/` に `BaseLLM` 参照が 1 件も無く、`ReportGenerateNode` は常に決定論レンダラを使っていた。
- `DemandAnomalyGraphNode._parent_config()` が `{}` を返しており、外側 config が内側サブグラフへ
  一切渡っていなかった（`config["llm"]` を注入しても届かない）。

修正内容:

1. 外側 graph が `_INNER_CONFIG_KEYS`（`llm` / `system_prompt`）を内側へ転送し、
   `ReportGenerateNode(llm=..., system_prompt=...)` として届くようにした（Retail Personalised Offer & Promo Copy Generator と同形）。
2. ナレーションのプロンプトを `src/services/llm_client.py` に置いた（テスト側には書かない）。
3. LLM 未注入時は決定論レンダラにフォールバックするが、`narration_mode` を出力に載せて
   **黙って決定論出力にならない**ようにした。
4. LLM の失敗・空応答・検知済み異常の記述漏れは `degradation_reason` として顕在化させ、
   決定論レポート（＝完全な表）へフォールバックする。

### 併せて強化した点（レビュー指摘反映）

| 項目 | 内容 |
|---|---|
| 接地ガード | `missing_true_anomaly_skus()` — true_anomaly の SKU がナレーションに現れない場合は公開せず決定論へ縮退。ASCII トークン境界照合（`\b` は日本語助詞で誤縮退するため不使用） |
| S-3（産出ノード側） | `ReportGenerateNode._extra_security_gate_output()` で PII を redact。外側 PostProcess だけでは、内側の生本文が State（checkpoint 有効時は checkpoint）を通過した後になる |
| S-1 | `ReportGenerateNode.required_trust_level` を `ANONYMOUS` → `VERIFIED_EXTERNAL`。課金対象の外部呼び出しを匿名で到達させない |
| 例外メッセージ | provider 例外は**型名のみ**を `degradation_reason` に載せる（本文は State に残さない） |

### 追加テスト

| テスト | 何を守るか |
|---|---|
| `tests/integration/test_end_to_end.py::TestLLMGenerationModeIsReal` | 外側 config の LLM が実 invoke 経路で `report_generate` に届くこと・プロンプトが検知結果を運ぶこと・未注入時のフォールバックが宣言されること・provider 失敗が縮退として caller に届くこと・`system_prompt` が届くこと |
| `tests/unit/test_services.py::TestLLMNarrationSeam` | LLM ナレーションの採用／空応答・失敗・記述漏れの縮退／CSV が決して LLM 由来にならないこと／SKU 照合の境界 |
| `tests/unit/test_nodes.py::TestReportGenerateNodeSecurityBoundary` | `node(state)`（`__call__` 経由）で PII redact と S-1 拒否（匿名では LLM を呼ばない）を証明 |

### 未解決（本 MR の範囲外・要判断）

- **standalone `src/api/server.py` は LLM を注入しない**（`RETDemandAnomalyAgent()` を無設定で構築）。
  本番でのLLM供給は gateway 側の `config["llm"]` 注入に依存する。Retail Personalised Offer & Promo Copy Generator のような
  `build_llm()` 遅延構築を入れるには `config/agent.yaml` の `requires.secrets` に
  `ANTHROPIC_API_KEY`、`requires.extras` に `anthropic` の宣言が必要だが、
  **`config/agent.yaml` は本作業の対象外**のため未対応（残件）。

## ローカル実行

```bash
source .venv/bin/activate
python -m pytest tests/ -v
python -m pytest tests/proof_of_boundary/ -v
```
