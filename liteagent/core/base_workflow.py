"""Base Workflow engine for LiteAgent framework.

Workflows are the top layer of the Skill→Agent→Workflow architecture.
They encapsulate the business process as a deterministic state graph
powered by LangGraph.

Each Workflow:
  - Defines a Pydantic Global State model (all fields have defaults).
  - Adds Agent nodes with explicit input_mapper and reducer functions.
  - Uses conditional edges based on deterministic State fields for routing.
  - Compiles to a runnable graph via LangGraph's compile().

Layer: Workflow layer (third layer).
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel

from liteagent.core.base_agent import AgentOutput, BaseAgent

_log = logging.getLogger(__name__)

StateType = TypeVar("StateType", bound=BaseModel)

InputMapperFn = Callable[[StateType], dict]
ReducerFn = Callable[[StateType, AgentOutput], dict]
ConditionFn = Callable[[StateType], str]


class BaseWorkflow(ABC):
    """Abstract base class for all Workflows.

    Wraps a LangGraph StateGraph and provides a controlled API for
    adding agent nodes with mandatory mapper/reducer contracts.

    Usage::

        class OrderState(BaseModel):
            order_id: str = ""
            search_result: str = ""
            risk_score: int = 0

        wf = OrderWorkflow(llm_adapter, skill_registry)
        result = wf.invoke(OrderState(order_id="ORD-001"))
    """

    def __init__(self, state_model: type[StateType]):
        """Initialize the workflow with a Pydantic state model.

        Args:
            state_model: A Pydantic BaseModel subclass used as the Global State.
        """
        self._state_model = state_model
        self._node_configs: dict[str, dict] = {}
        self._edges: list[tuple[str, str]] = []
        self._conditional_edges: list[dict] = []
        self._entry_point: Optional[str] = None
        self._compiled = False

        self._graph = self._create_graph()

    def _create_graph(self):
        """Create a LangGraph StateGraph with the state model.

        Returns:
            A configured StateGraph instance.
        """
        from langgraph.graph import StateGraph

        return StateGraph(self._state_model)

    def set_entry_point(self, node_name: str) -> None:
        """Set the entry point node for the graph.

        Args:
            node_name: Name of the starting node.
        """
        self._entry_point = node_name
        self._graph.set_entry_point(node_name)

    def set_finish_point(self, node_name: str) -> None:
        """Set a finish/end point for the graph.

        Args:
            node_name: Name of the terminal node.
        """
        self._graph.set_finish_point(node_name)

    def add_agent_node(
        self,
        name: str,
        agent: BaseAgent,
        input_mapper: InputMapperFn,
        reducer: ReducerFn,
    ) -> None:
        """Add a node that invokes an Agent with explicit mapper/reducer contracts.

        The mapper extracts only the needed fields from Global State into a
        local dict for the agent. The reducer merges the AgentOutput back
        into the Global State.

        Args:
            name: Unique node name within the graph.
            agent: The BaseAgent instance to invoke.
            input_mapper: Function (GlobalState) -> local_dict.
            reducer: Function (GlobalState, AgentOutput) -> partial_update_dict.
        """
        if self._compiled:
            raise RuntimeError("Cannot add nodes after the graph has been compiled.")

        # Validate with Pydantic v2 that the state model fields are all defaulted
        for field_name, field_info in self._state_model.model_fields.items():
            if field_info.is_required():
                raise ValueError(
                    f"Global State field '{field_name}' in {self._state_model.__name__} "
                    f"must have a default value. All fields on Global State models require defaults."
                )

        self._node_configs[name] = {
            "agent": agent,
            "input_mapper": input_mapper,
            "reducer": reducer,
        }

        def node_fn(state: StateType) -> dict:
            _log.info("node_enter", extra={"layer": "workflow", "node_name": name})
            try:
                local_input = input_mapper(state)
                agent_output = agent.invoke(local_input)
                partial_update = reducer(state, agent_output)
                _log.info("node_done", extra={"layer": "workflow", "node_name": name})
                return partial_update
            except Exception as e:
                _log.error("node_failed", extra={"layer": "workflow", "node_name": name, "error": str(e)}, exc_info=True)
                return {
                    "error": f"Agent node '{name}' failed: {e}",
                }

        self._graph.add_node(name, node_fn)

    def add_edge(self, source: str, target: str) -> None:
        """Add a direct edge between two nodes.

        Args:
            source: Source node name.
            target: Target node name.
        """
        if self._compiled:
            raise RuntimeError("Cannot add edges after the graph has been compiled.")
        self._edges.append((source, target))
        self._graph.add_edge(source, target)

    def add_conditional_edges(
        self,
        source: str,
        condition_fn: ConditionFn,
        route_map: dict[str, str],
    ) -> None:
        """Add conditional routing from a source node.

        The condition_fn inspects the Global State and returns a string
        that maps to the next node via route_map.

        Args:
            source: Source node name.
            condition_fn: Function (GlobalState) -> str returning a routing key.
            route_map: Dict mapping routing keys to target node names.
        """
        if self._compiled:
            raise RuntimeError("Cannot add conditional edges after the graph has been compiled.")

        self._conditional_edges.append({
            "source": source,
            "condition_fn": condition_fn,
            "route_map": route_map,
        })
        self._graph.add_conditional_edges(source, condition_fn, route_map)

    def compile(self) -> Any:
        """Compile the graph into a runnable application.

        Returns:
            A compiled LangGraph application ready for invoke().
        """
        if self._entry_point is None:
            raise ValueError("Entry point not set. Call set_entry_point() before compiling.")
        self._compiled = True
        return self._graph.compile()

    def invoke(self, initial_state: StateType, compiled: Any = None) -> StateType:
        """Run the workflow from an initial state.

        Compiles the graph on first call or uses a pre-compiled graph.

        Args:
            initial_state: The starting Global State instance.
            compiled: Optional pre-compiled graph from compile().

        Returns:
            The final Global State after the workflow completes.
        """
        app = compiled or self.compile()
        _log.info("workflow_start", extra={"layer": "workflow", "state_model": self._state_model.__name__})

        result = app.invoke(initial_state)

        _log.info("workflow_done", extra={"layer": "workflow"})

        if isinstance(result, dict):
            return self._state_model(**result)
        return result
