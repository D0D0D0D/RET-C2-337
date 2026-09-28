"""AgentCore Platform v1.0 — RET-C2-337 outer graph (Cat 2).

Retail Demand Anomaly Detection Agent.

Cat 2 two-layer composition:
  - Outer: RETDemandAnomalyAgent(AgentBaseGraph) — fixed 5-node backbone.
  - main slot: DemandAnomalyGraphNode(GraphNode) wrapping the inner
    DemandAnomalyWorkflowGraph (5-node detection workflow).

Detection (data_normalize -> covariate_forecast -> stat_anomaly_detect ->
event_attribution) is deterministic Tool calling; the LLM is restricted to the
report_generate step. add_edges() is NOT overridden — backbone wiring belongs
to the framework.
"""

from __future__ import annotations

from typing import ClassVar

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState

from src.nodes.pre_process_node import PreProcessNode
from src.nodes.post_process_node import PostProcessNode
from src.schemas.state import State


# Config keys forwarded from the outer graph into the inner workflow.
# `llm` is the narration seam for report_generate (docs/02_design.md §3).
_INNER_CONFIG_KEYS = ("llm", "system_prompt")


class DemandAnomalyGraphNode(GraphNode):
    """`main`-slot GraphNode: runs the inner demand-anomaly workflow."""

    error_strategy: ClassVar[str] = "propagate"
    propagate_hitl: ClassVar[bool] = False

    def __init__(self, inner_config: dict | None = None) -> None:
        super().__init__()
        # Static configuration only — never per-invocation state (node instances
        # are shared across every invocation).
        self._inner_config = dict(inner_config or {})

    def get_subgraph(self):
        from src.graph.domain_workflow_graph import DemandAnomalyWorkflowGraph

        return DemandAnomalyWorkflowGraph(config=self._parent_config())

    def extract_input(self, state: AgentState) -> str:
        # The inner graph receives only a string (BaseGraph.invoke contract), so
        # pass the parsed request as a JSON string; the inner data_normalize node
        # parses it back. Falls back to the raw validated_input if unparsed.
        import json

        parsed = state.get("parsed_request")
        if parsed:
            return json.dumps(parsed)
        return state.get("validated_input", state.get("user_input", ""))

    def merge_output(self, state: AgentState, sub_result: dict) -> dict:
        """Bundle inner result into a single `anomaly_result` field."""
        return {
            "anomaly_result": {
                "attributed_anomalies": sub_result.get("attributed_anomalies", []),
                "report_markdown": sub_result.get("report_markdown", ""),
                "report_csv": sub_result.get("report_csv", ""),
                "narration_mode": sub_result.get("report_narration_mode", ""),
                "degradation_reason": sub_result.get("report_degradation_reason", ""),
            },
            "status": sub_result.get("status"),
        }

    def _parent_config(self) -> dict:
        return self._inner_config


class RETDemandAnomalyAgent(AgentBaseGraph):
    """Cat 2 outer graph — Retail Demand Anomaly Detection Agent."""

    @property
    def name(self) -> str:
        return "ret_c2_337"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        super().register_nodes()  # injects initialize + finalize
        inner_config = {k: self.config[k] for k in _INNER_CONFIG_KEYS if k in self.config}
        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = DemandAnomalyGraphNode(inner_config=inner_config)
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden — backbone wiring belongs to the framework.
