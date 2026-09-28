"""AgentCore Platform v1.0 — inner domain workflow graph for RET-C2-337.

DemandAnomalyWorkflowGraph wraps the 5-node detection workflow:
    data_normalize -> covariate_forecast -> stat_anomaly_detect
        -> event_attribution -> report_generate

Inherits BaseGraph (fully custom topology). Instantiated by
DemandAnomalyGraphNode.get_subgraph() in graph.py. get_output() shapes the
sub_result consumed by the outer merge_output().
"""

from __future__ import annotations

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_status import AgentStatus

from src.nodes.data_normalize_node import DataNormalizeNode
from src.nodes.covariate_forecast_node import CovariateForecastNode
from src.nodes.stat_anomaly_detect_node import StatAnomalyDetectNode
from src.nodes.event_attribution_node import EventAttributionNode
from src.nodes.report_generate_node import ReportGenerateNode
from src.schemas.state import State


class DemandAnomalyWorkflowGraph(BaseGraph):
    """Inner 5-node detection workflow (linear topology)."""

    @property
    def name(self) -> str:
        return "demand_anomaly_workflow"

    @property
    def state_schema(self) -> type:
        return State

    def _validate_config(self) -> None:
        # No mandatory config; inference_mode flows via parsed_request in state.
        pass

    def register_nodes(self) -> None:
        # No super() call — BaseGraph.register_nodes() is abstract.
        # Do NOT register initialize/finalize (outer backbone concern).
        self._nodes["data_normalize"] = DataNormalizeNode()
        self._nodes["covariate_forecast"] = CovariateForecastNode()
        self._nodes["stat_anomaly_detect"] = StatAnomalyDetectNode()
        self._nodes["event_attribution"] = EventAttributionNode()
        # LLM seam: the outer graph forwards config["llm"] here (docs/02_design.md §3 —
        # report_generate is the only LLM step). Absent -> deterministic narration.
        self._nodes["report_generate"] = ReportGenerateNode(
            llm=self.config.get("llm"),
            system_prompt=self.config.get("system_prompt"),
        )

    def add_edges(self) -> None:
        self._sg.add_edge(START, "data_normalize")
        self._sg.add_edge("data_normalize", "covariate_forecast")
        self._sg.add_edge("covariate_forecast", "stat_anomaly_detect")
        self._sg.add_edge("stat_anomaly_detect", "event_attribution")
        self._sg.add_edge("event_attribution", "report_generate")
        self._sg.add_edge("report_generate", END)

    def route(self, state) -> str:
        # Linear topology — route() is required by the ABC but never invoked
        # unless add_conditional_edges() references it.
        return END if state.get("status") == AgentStatus.ERROR.value else "report_generate"

    def get_output(self, state) -> dict:
        return {
            "attributed_anomalies": state.get("attributed_anomalies", []),
            "report_markdown": state.get("report_markdown", ""),
            "report_csv": state.get("report_csv", ""),
            "report_narration_mode": state.get("report_narration_mode", ""),
            "report_degradation_reason": state.get("report_degradation_reason", ""),
            "status": state.get("status"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
