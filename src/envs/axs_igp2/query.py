"""Query implementation for IGP2."""

from typing import Any, ClassVar

import igp2 as ip
from shapely import Polygon

import axs

from .macroaction import IGP2MacroAction


class IGP2Query(axs.Query):
    """Query implementation for IGP2."""

    args_and_types: ClassVar[dict[str, dict[str, type]]] = {
        "add": {"location": tuple[float, float], "goal": tuple[float, float]},
        "remove": {"vehicle": int},
        "whatif": {"vehicle": int, "actions": list[IGP2MacroAction], "time": int},
        "what": {"vehicle": int, "time": int},
    }

    def __init__(self, name: str, params: dict[str, Any]) -> "IGP2Query":
        """Initialize new Query.

        Args:
            name (str): The name of the query.
            params (dict): The parameters for the query.

        """
        super().__init__(name, params)
        if name == "remove" and params["vehicle"] == 0:
            error_msg = "Cannot remove the ego vehicle."
            raise axs.QueryError(error_msg)

    def verify(
        self,
        env: ip.simplesim.SimulationEnv | None = None,
        observations: list[Any] | None = None,
        actions: list[Any] | None = None,
        macro_actions: dict[str, list[IGP2MacroAction]] | None = None,
        infos: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Verify the query is valid.

        Args:
            env (ip.simplesim.SimulationEnv): The simulation environment.
            observations (Any): The observations from the environment.
            actions (Any): The actions taken by the agent.
            macro_actions (dict[str, list[IGP2MacroAction]]): Macros for each vehicle.
            infos (dict[str, Any]): Additional informations about the simulation.

        """
        if self.query_name == "remove":
            vid = self.params["vehicle"]
            if vid == 0:
                error_msg = "Cannot remove the ego vehicle."
                raise axs.QueryError(error_msg)
            if vid not in infos[0]:
                error_msg = f"Vehicle {vid} does not exist."
                raise axs.QueryError(error_msg)

        if self.query_name == "add":
            location = self.params["location"]
            goal = self.params["goal"]
            spawn_box = Polygon(ip.Box(location, 1, 1, 0.0).boundary)
            goal_box = Polygon(ip.Box(goal, 1, 1, 0.0).boundary)
            spawn_intersects = False
            goal_intersects = False
            for road in env.scenario_map.roads.values():
                if road.boundary.intersects(spawn_box):
                    spawn_intersects = True
                if road.boundary.intersects(goal_box):
                    goal_intersects = True
            if not spawn_intersects or not goal_intersects:
                error_msg = "Spawn and goal locations must be on a road."
                raise axs.QueryError(error_msg)

            for state in infos[0].values():
                vehicle_box = Polygon(
                    ip.Box(
                        state.position,
                        state.metadata.length,
                        state.metadata.width,
                        state.heading,
                    ).boundary,
                )
                if spawn_box.intersects(vehicle_box):
                    error_msg = "Spawn location intersects with another vehicle."
                    raise axs.QueryError(error_msg)

        if self.query_name == "whatif":
            vehicle = self.params["vehicle"]
            time = min(max(0, len(infos) - 1), self.get_time(infos[-1][vehicle].time))
            if vehicle not in infos[time]:
                error_msg = f"Vehicle {vehicle} does not exist."
                raise axs.QueryError(error_msg)
            actions = self.params["actions"]
            macro_at_time = next(
                ma for ma in macro_actions[vehicle] if ma.start_t <= time <= ma.end_t
            ).macro_name
            if len(actions) == 1 and actions[0].macro_name == macro_at_time:
                error_msg = "Cannot test the same action as the last action."
                raise axs.QueryError(error_msg)

        if self.query_name == "what":
            vehicle = self.params["vehicle"]
            if vehicle not in infos[-1]:
                error_msg = f"Vehicle {vehicle} does not exist."
                raise axs.QueryError(error_msg)
            self.get_time(0)  # Check time parameter

        return True

    def get_time(self, current_time: int) -> int:
        """Return the time parameter of the query."""
        time = self.params.get("time", None)

        if time is None:
            if self.query_name in ["add", "remove"]:
                time = 0
            elif self.query_name == "what":
                time = current_time
            else:
                error_msg = f"Time parameter not found for query: {self.query_name}"
                raise axs.QueryError(error_msg)
        elif time < 0:
            error_msg = f"Time parameter cannot be negative: {time}"
            raise axs.QueryError(error_msg)
        elif time > current_time and self.query_name != "what":
            error_msg = f"Time {time} is in the future for query type {self.query_name}"
            raise axs.QueryError(error_msg)

        return time

    @classmethod
    def query_descriptions(cls) -> dict[str, str]:
        """Return a string with the query descriptions.

        You may refer query variables in your description using the format <variable>.
        These descriptions are used to generate the user prompt for the LLM.
        """
        return {
            "add": "What would happen if a new vehicle was present at $location with $goal from the start?",  # noqa: E501
            "remove": "What would happen if $vehicle was removed from the road?",
            "whatif": "What would happen if $vehicle took $actions starting from $time?",  # noqa: E501
            "what": "What will $vehicle be doing at $time?",
        }

    @classmethod
    def query_type_descriptions(cls) -> dict[str, str]:
        """Return a string with the query type descriptions.

        If not overriden, then descriptions are generated from the args_and_types
        automatically. The descriptions are used to generate user prompts for the LLM.
        """
        return {
            "location": "2D coordinate",
            "goal": "2D coordinate",
            "vehicle": "int",
            "actions": "list of macro action names from {macro_names}.",
            "time": "integer",
        }
