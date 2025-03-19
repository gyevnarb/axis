"""Verbalize IGP2 simulation data."""

import logging
from collections import defaultdict
from typing import Any

import igp2 as ip
import numpy as np
from igp2.opendrive.elements.geometry import ramer_douglas

import axs
from envs.axs_igp2 import util
from envs.axs_igp2.macroaction import IGP2MacroAction

logger = logging.getLogger(__name__)


ROAD_LAYOUT_PRETEXT = """Below metadata to parse the elements of the road layout.

Coordinate system:
  We are using a 2D Cartesian coordinate system.
  Coordinates are in units of meters and written as (x, y).
  Angles are in radians in the range [-pi, pi].

Roads and Lanes:
  The road layout consists of roads which are given a unique numeric ID.
  Roads are made up of lanes which are identified as 'road ID:lane ID'.
  Lanes are oriented in the direction of the road midline.
  Lanes are divided into left and right lanes.
  Right lanes have a negative ID and left lanes have a positive ID.
  Lanes are 3.5 meters wide.

Intersections:
  Roads are connected at intersections.
  Intersections are made up of connections between incoming and connecting roads.
"""


class IGP2Verbalizer(axs.Verbalizer):
    """Verbalize the environment, observations, and state for IGP2."""

    @staticmethod
    def convert(
        env: ip.simplesim.SimulationEnv,
        observations: list[np.ndarray],  # noqa: ARG004
        macro_actions: dict[int, list[IGP2MacroAction]],
        infos: list[dict[str, Any]],
        **kwargs,
    ) -> str:
        """Verbalize the IGP2 scenario.

        Args:
            env (ip.simplesim.SimulationEnv): The IGP2 environment.
            observations (list): The observations of the agents. Not used.
            macro_actions (list): The macro actions of the agents.
            infos (list): The information of the agents. Not used.
            kwargs: Optional keyword arguments.
                - add_roads: Whether to add road descriptions.
                - add_macro_actions: Whether to add macro action descriptions.
                - add_observations: Whether to add observation descriptions.
                - add_infos: Whether to add agent information.
                - f_subsample (int): Frequency of subsampling observations.
                        Use this to decrease the complexity of the verbalization.
                - rounding (int): Number of decimal places to round the values to.
                - control_signals (list[str]): List of control signals to include.
                        Possible values: ["times", "timesteps", "path", "velocity",
                            "acceleration", "heading", "angular_velocity"].
                        Default is all control signals except time.
                - add_lanes: Whether to add lane descriptions (True).
                - add_intersections: Whether to add intersection descriptions (True).
                - add_intersection_links: Whether to add intersection lane link (False).
                - resolution: The resolution of the road midline (0.01).
                - add_metadata: Whether to add metadata before
                            the road layout description (False).

        """
        context = ""
        if kwargs.get("add_roads", False):
            context += IGP2Verbalizer.convert_environment(env, kwargs) + "\n\n"

        actions_dict = IGP2Verbalizer._convert_macro_actions(macro_actions)
        infos_dict = IGP2Verbalizer._convert_infos(infos, **kwargs)
        if set(actions_dict.keys()) != set(infos_dict.keys()):
            error_msg = "Agent IDs in actions and infos do not match."
            raise ValueError(error_msg)

        for agent_id in actions_dict:
            context += f"Vehicle {agent_id}:\n"
            if kwargs.get("add_infos", True):
                context += "  Observations:\n"
                for signal in infos_dict[agent_id]:
                    context += f"    {signal}"

            if kwargs.get("add_macro_actions", True):
                context += "  Actions:\n"
                for segment in actions_dict[agent_id]:
                    context += f"    {segment}\n"

            context += "\n"

        return context[:-1]  # Remove trailing newline

    @staticmethod
    def convert_infos(infos: list[dict[str, Any]], **kwargs) -> str:
        """Verbalize a frame of the simulation state.

        Args:
            infos (list[str, dict[Any]]): Sequence of info dictionaries for each agent.
            kwargs: Optional keyword arguments.
                - f_subsample (int): Frequency of subsampling observations.
                        Use this to decrease the complexity of the verbalization.
                - rounding (int): Number of decimal places to round the values to.
                - control_signals (list[str]): List of control signals to include.
                        Possible values: ["times", "timesteps", "path", "velocity",
                            "acceleration", "heading", "angular_velocity"].
                        Default is all control signals except time.

        """
        ret = "Observations:\n"
        infos_dict = IGP2Verbalizer._convert_infos(infos, **kwargs)
        for agent_id, control_signals in infos_dict.items():
            ret += f"  Vehicle {agent_id}:\n"
            for signal in control_signals:
                ret += f"    {signal}\n"
            ret += "\n"
        return ret[:-1]

    @staticmethod
    def _convert_infos(
        infos: dict[int, list[IGP2MacroAction]], **kwargs,
    ) -> list[str]:
        trajectories = defaultdict(list)
        for frame in infos:
            for agent_id, state in frame.items():
                trajectories[agent_id].append(state)
        trajectories = {k: ip.StateTrajectory(None, v) for k, v in trajectories.items()}

        ret = {}

        f_subsample = kwargs.get("f_subsample", 1)
        rounding = kwargs.get("rounding", 2)
        control_signals = kwargs.get(
            "control_signals",
            ["timesteps", "path", "velocity", "acceleration", "angular_velocity"],
        )
        for agent_id, trajectory in trajectories.items():
            sampled_trajectory = trajectory
            if f_subsample > 1:
                sampled_trajectory = util.subsample_trajectory(trajectory, f_subsample)

            ret[agent_id] = [
                IGP2Verbalizer._verbalize_control_signal(
                    signal,
                    rounding,
                    sampled_trajectory,
                )
                for signal in control_signals
            ]

        return ret

    @staticmethod
    def convert_observations(observations: list[Any], **kwargs) -> str:  # noqa: ARG004
        """Verbalize the observations of the agents. Not used in IGP2."""
        logger.debug("IGP2 does not use Verbalizer.convert_observations.")
        return ""

    @staticmethod
    def convert_macro_actions(
        macro_actions: dict[int, list[IGP2MacroAction]],
        **kwargs,  # noqa: ARG004
    ) -> str:
        """Verbalize the macro actions of the agents."""
        ret = "Actions:\n"
        segments_dict = IGP2Verbalizer._convert_macro_actions(macro_actions)
        for agent_id, segments_str in segments_dict.items():
            ret += f"Vehicle {agent_id}: {segments_str}\n"
        return ret

    @staticmethod
    def _convert_macro_actions(
        macro_actions: dict[int, list[IGP2MacroAction]],
    ) -> dict[int, str]:
        ret = {}
        for agent_id, segmentations in macro_actions.items():
            ret[agent_id] = ", ".join(map(repr, segmentations))
        return ret

    @staticmethod
    def convert_environment(env: ip.simplesim.SimulationEnv, **kwargs) -> str:
        """Verbalize the road layout.

        Args:
            env (ip.simplesim.SimulationEnv): The igp2 environment to verbalize.
            kwargs: Optional keyword arguments.
                - add_lanes: Whether to add lane descriptions (True).
                - add_intersections: Whether to add intersection descriptions (True).
                - add_intersection_links: Whether to add intersection lane link (False).
                - resolution: The resolution of the road midline (0.01).
                - add_metadata: Whether to add metadata before
                            the road layout description (False).

        Returns:
            A string describing the road layout.

        """
        scenario_map = env.scenario_map
        ret = ""

        add_metadata = kwargs.get("add_metadata", False)
        if add_metadata:
            ret += ROAD_LAYOUT_PRETEXT
            lane_links = kwargs.get("intersection_links", False)
            ret += "  Connections are written as "
            if not lane_links:
                ret += "'incoming road id->connecting road id'."
            else:
                ret += "'incoming road id:lane id->connecting road id:lane id'."
            ret += "\n\n"

        ret += "The road layout consists of the following elements:"
        ret += "\n\n"

        # Describe roads
        IGP2Verbalizer._add_verbalized_roads(ret, scenario_map, kwargs)

        # Describe intersections
        if kwargs.get("add_intersections", True):
            for jid, junction in scenario_map.junctions.items():
                ret += f"Intersection {jid} connections:\n"
                for conn in junction.connections:
                    if kwargs.get("add_intersection_links", False):
                        for lane_link in conn.lane_links:
                            ret += f"  {conn.incoming_road.id}.{lane_link.from_id}"
                            ret += f"->{conn.connecting_road.id}.{lane_link.to_id}\n"
                    else:
                        ret += f"  {conn.incoming_road.id}->{conn.connecting_road.id}\n"

        if ret[-1] == "\n":
            ret = ret[:-1]
        return ret

    @staticmethod
    def _add_verbalized_roads(
        ret: str, scenario_map: ip.Map, kwargs: dict[str, Any],
    ) -> None:
        for rid, road in scenario_map.roads.items():
            if not road.drivable:
                continue

            ret += f"Road {rid}:\n"
            ret += f"  Length: {road.length} m\n"

            midline = ramer_douglas(
                np.array(road.midline.coords), dist=kwargs.get("resolution", 0.02),
            )
            midline = [(x, y) for x, y in np.round(midline, 2)]
            ret += f"  Midline coordinates: {midline}\n"

            left_lanes = [
                lane
                for lane in road.lanes.lane_sections[0].left_lanes
                if lane.type == ip.LaneTypes.DRIVING
            ]
            right_lanes = [
                lane
                for lane in road.lanes.lane_sections[0].right_lanes
                if lane.type == ip.LaneTypes.DRIVING
            ]

            # Describe lanes
            if kwargs.get("add_lanes", True):
                if left_lanes:
                    ret += "  Left lanes:\n"
                    for lane in left_lanes:
                        ret += f"    Lane {rid}.{lane.id}.\n"
                if right_lanes:
                    ret += "  Right lanes:\n"
                    for lane in right_lanes:
                        ret += f"    Lane {rid}.{lane.id}.\n"
            ret += "\n"

    @staticmethod
    def _verbalize_control_signal(
        signal: str,
        precision: int,
        trajectory: ip.StateTrajectory,
    ) -> str:
        if signal == "timesteps":
            timesteps = np.array([s.time for s in trajectory.states])
            return f"Timesteps: {util.ndarray2str(timesteps)}\n"
        if signal == "maneuver":
            mans = [s.maneuver for s in trajectory.states]
            return f"Maneuver: {mans}\n"
        if signal == "macro":
            macros = [s.macro_action for s in trajectory.states]
            return f"Macro action: {macros}\n"
        if hasattr(trajectory, signal):
            name = {
                "times": "Time",
                "path": "Position",
                "velocity": "Speed",
                "acceleration": "Acceleration",
                "heading": "Heading",
                "angular_velocity": "Steering",
            }[signal]
            txt_signal = util.ndarray2str(getattr(trajectory, signal), precision)
            return f"{name}: {txt_signal}\n"
        error_msg = f"Unknown control signal: {signal}"
        raise ValueError(error_msg)
