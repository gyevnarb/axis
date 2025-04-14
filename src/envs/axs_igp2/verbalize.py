"""Verbalize IGP2 simulation data."""

import logging
from collections import defaultdict
from typing import Any

import gofi
import igp2 as ip
import numpy as np
from igp2.opendrive.elements.geometry import ramer_douglas

import axs
from envs.axs_igp2 import util
from envs.axs_igp2.macroaction import IGP2MacroAction

logger = logging.getLogger(__name__)


ROAD_LAYOUT_PRETEXT = """- Metadata:
  - Traffic rules:
    - Vehicles drive on the right side of the road.
    - The maximum speed limit is 10 m/s.
  - Coordinate system:
    - Coordinates are on a 2D Cartesian plane.
    - Coordinates are written as [x y].
    - Distances are in meters.
    - Angles are in radians in the range [-pi, pi].
  - Road layout:
    - The road layout consists of roads identified by a numeric ID.
    - Roads are made up of lanes identified as road ID:lane ID.
    - Lanes are divided into left and right lanes.
    - Right lanes have a negative ID and left lanes have a positive ID.
    - Lanes are 3.5 meters wide.
  - Intersections:
    - Roads are connected by intersections.
    - Intersections are made up of connections between incoming and connecting roads.
"""

REWARD_NAME_MAP = {
    "jerk": "Jolt",
    "coll": "Collision",
    "curvature": "Curvature",
    "term": "Goal-not-reached",
    "angular_velocity": "Steering",
    "time": "Time-to-Goal",
    "dead": "Goal-not-Reached",
}


class IGP2Verbalizer(axs.Verbalizer):
    """Verbalize the environment, observations, and state for IGP2."""

    @staticmethod
    def convert(
        observations: list[np.ndarray],  # noqa: ARG004
        macro_actions: dict[int, list[IGP2MacroAction]],
        infos: list[dict[str, ip.AgentState]] | None = None,
        rewards: dict[int, ip.Reward] | None = None,
        query: axs.Query | None = None,
        env: ip.simplesim.SimulationEnv | None = None,
        **kwargs: dict[str, Any],
    ) -> dict[str, str]:
        """Verbalize the IGP2 scenario.

        Args:
            observations (list): The observations of the agents. Not used.
            macro_actions (list): The macro actions of the agents.
            infos (list): The information of the agents.
            rewards (dict[str, float] | None): Any rewards to verbalize.
            query (axs.Query | None): The query to verbalize.
            env (ip.simplesim.SimulationEnv | None): The IGP2 environment.
            kwargs: Optional keyword arguments.
                - add_layout: Whether to add the road layout description.
                - add_roads: Whether to add road descriptions.
                - add_metadata: Whether to add metadata before
                            the road layout description.
                - add_buildings: Whether to add building descriptions.
                - add_actions: Whether to add raw steering and acceleration values.
                - add_macro_actions: Whether to add macro action descriptions.
                - add_observations: Whether to add observation descriptions.
                - add_rewards: Whether to add reward descriptions.
                - add_lanes: Whether to add lane descriptions.
                - add_intersections: Whether to add intersection descriptions.
                - add_intersection_links: Whether to add intersection lane link.
                - subsample (int): Frequency of subsampling observations.
                        Use this to decrease the complexity of the verbalization.
                - rounding (int): Number of decimal places to round the values to.
                - state_signals (list[str]): List of control signals to include.
                        Possible values: ["times", "timesteps", "path",
                                          "velocity", "heading"].
                        Default is all control signals except time.
                - resolution: The resolution of the road midline (0.01).
                - fps: The frames per second of the simulation (20).

        Returns:
            context (dict[str, str]): Dictionary of verbalized data with keys mapping to
                argument names in a axs.Query objetc

        """
        if not isinstance(macro_actions, dict) and not all(
            isinstance(k, int) for k in macro_actions
        ):
            error_msg = (
                f"Macro actions must be a dictionary with "
                f"int agent ids as keys. Got: {macro_actions}"
            )
            raise ValueError(error_msg)

        if isinstance(rewards, list) and "reward" in infos[-1]:
            rewards = {0: infos[-1].pop("reward")}

        context = ""
        ret = {}

        if kwargs.get("add_layout", False):
            context += IGP2Verbalizer.convert_environment(env, **kwargs) + "\n\n"

        actions_dict = IGP2Verbalizer._convert_macro_actions(macro_actions)
        infos_dict = IGP2Verbalizer._convert_infos(infos, **kwargs)
        if set(actions_dict.keys()) != set(infos_dict.keys()):
            error_msg = "Agent IDs in actions and infos do not match."
            raise ValueError(error_msg)

        add_observations = kwargs.get("add_observations", False)
        add_macro_actions = kwargs.get("add_macro_actions", False)
        add_actions = kwargs.get("add_actions", False)
        add_rewards = kwargs.get("add_rewards", False)

        for aid in actions_dict:
            context += f"- Vehicle {aid}:\n"
            if add_observations or add_actions:
                context += "  - Observations:\n"
            if add_observations:
                for signal, data in infos_dict[aid].items():
                    if signal in ["Steering", "Acceleration"]:
                        continue  # Do not include actions here
                    context += f"    - {signal}: {data}\n"
            if add_actions:
                # context += "  - Actions:\n"
                if not add_observations:
                    context += f"    - Timesteps: {infos_dict[aid]['Timesteps']}\n"
                context += f"    - Steering: {infos_dict[aid]['Steering']}\n"
                context += f"    - Acceleration: {infos_dict[aid]['Acceleration']}\n"
            if add_macro_actions:
                context += "  - Macro actions (as macro[from-to]): "
                context += f"[{actions_dict[aid]}]\n"
            if (
                add_rewards
                and rewards is not None
                and isinstance(rewards, dict)
                and aid in rewards
            ):
                context += "  - Rewards:\n"
                reward_str = IGP2Verbalizer._convert_reward(rewards[aid], **kwargs)
                context += f"{reward_str}\n"
            context += "\n"
        context = context[:-2]  # Remove trailing newlines

        ret["context"] = context

        if query is not None:
            q_descriptions, q_type_descriptions = IGP2Verbalizer.convert_query(query)
            ret["query_descriptions"] = q_descriptions
            ret["query_type_descriptions"] = q_type_descriptions

        return ret

    @staticmethod
    def convert_rewards(rewards: ip.Reward, **kwargs: dict[str, Any]) -> str:
        """Verbalize the rewards of the agents.

        Args:
            rewards (dict[str, float] | None): Any rewards to verbalize.
            kwargs: Optional keyword arguments.

        """
        ret = "Rewards:\n"
        for agent_id, reward in rewards.items():
            reward_str = IGP2Verbalizer._verbalize_reward(reward, **kwargs)
            ret += f"  Vehicle {agent_id}: {reward_str}\n"
        return ret[:-1]

    @staticmethod
    def _convert_reward(reward: ip.Reward, **kwargs: dict[str, Any]) -> str:
        """Verbalize the IGP2 reward class of an agent.

        Args:
            reward (dict[str, float]): The reward to verbalize.
            kwargs: Optional keyword arguments.
                - rounding (int): Number of decimal places to round the values to.
                - exclude_rewards (list[str]): List of reward signals to exclude.

        """
        ret = ""
        for key, value in reward.reward_components.items():
            if key in kwargs.get("exclude_rewards", []) or value is None:
                continue
            reward_name = REWARD_NAME_MAP.get(key, key)
            rounded_value = np.round(value, kwargs.get("rounding", 3))
            ret += f"    - {reward_name}: {rounded_value}\n"
        return ret[:-1]

    @staticmethod
    def convert_infos(infos: list[dict[str, Any]], **kwargs: dict[str, Any]) -> str:
        """Verbalize a frame of the simulation state.

        Args:
            infos (list[str, dict[Any]]): Sequence of info dictionaries for each agent.
            kwargs: Optional keyword arguments.
                - subsample (int): Frequency of subsampling observations.
                        Use this to decrease the complexity of the verbalization.
                - rounding (int): Number of decimal places to round the values to.
                - state_signals (list[str]): List of control signals to include.
                        Possible values: ["times", "timesteps", "path", "velocity",
                            "acceleration", "heading", "angular_velocity"].
                        Default is all control signals except time.

        """
        ret = "Observations:\n"
        infos_dict = IGP2Verbalizer._convert_infos(infos, **kwargs)
        for agent_id, state_signals in infos_dict.items():
            ret += f"  Vehicle {agent_id}:\n"
            for signal, data in state_signals.items():
                ret += f"    {signal}: {data}\n"
            ret += "\n"
        return ret[:-1]

    @staticmethod
    def _convert_infos(
        infos: dict[int, list[IGP2MacroAction]],
        **kwargs: dict[str, Any],
    ) -> dict[int, dict[str, str]]:
        trajectories = defaultdict(list)
        for frame in infos:
            for agent_id, state in frame.items():
                trajectories[agent_id].append(state)

        trajectories = {
            k: ip.StateTrajectory(kwargs.get("fps", 20), v)
            for k, v in trajectories.items()
        }

        ret = {}

        subsample = kwargs.get("subsample", 1)
        rounding = kwargs.get("rounding", 2)
        state_signals = kwargs.get(
            "state_signals",
            ["timesteps", "path", "velocity"],
        )
        # We always calculate these as they may be included as part of the actions
        state_signals.extend(["angular_velocity", "acceleration"])
        for agent_id, trajectory in trajectories.items():
            sampled_trajectory = trajectory
            if subsample > 1 and len(trajectory) > subsample:
                start_t = trajectory[0].time
                sampled_trajectory = util.subsample_trajectory(
                    trajectory,
                    start_t,
                    subsample,
                )

            ret[agent_id] = dict(
                [
                    IGP2Verbalizer._verbalize_control_signal(
                        signal,
                        rounding,
                        sampled_trajectory,
                    )
                    for signal in state_signals
                ],
            )

        return ret

    @staticmethod
    def convert_observations(observations: list[Any], **kwargs: dict[str, Any]) -> str:  # noqa: ARG004
        """Verbalize the observations of the agents. Not used in IGP2."""
        logger.debug("IGP2 does not use Verbalizer.convert_observations.")
        return ""

    @staticmethod
    def convert_macro_actions(
        macro_actions: dict[int, list[IGP2MacroAction]],
        **kwargs: dict[str, Any],  # noqa: ARG004
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
        **kwargs: dict[str, Any],  # noqa: ARG004
    ) -> dict[int, str]:
        ret = {}
        for agent_id, segmentations in macro_actions.items():
            ret[agent_id] = ", ".join(map(repr, segmentations))
        return ret

    @staticmethod
    def convert_environment(
        env: ip.simplesim.SimulationEnv,
        **kwargs: dict[str, Any],
    ) -> str:
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

        add_metadata = kwargs.get("add_metadata", True)
        if add_metadata:
            ret += ROAD_LAYOUT_PRETEXT
            lane_links = kwargs.get("intersection_links", False)
            ret += "    - Connections are written as "
            if not lane_links:
                ret += "incoming road id->connecting road id."
            else:
                ret += "incoming road id:lane id->connecting road id:lane id."
            ret += "\n\n"

        # Describe roads
        if kwargs.get("add_roads", True):
            ret += "- Road layout:\n"
            ret = IGP2Verbalizer._add_verbalized_roads(ret, scenario_map, **kwargs)

        # Describe intersections
        if kwargs.get("add_intersections", True):
            for jid, junction in scenario_map.junctions.items():
                ret += f"  - Intersection {jid}:\n"
                for conn in junction.connections:
                    if kwargs.get("add_intersection_links", False):
                        for lane_link in conn.lane_links:
                            ret += f"    - {conn.incoming_road.id}:{lane_link.from_id}"
                            ret += f"->{conn.connecting_road.id}:{lane_link.to_id}\n"
                    else:
                        ret += f"    - {conn.incoming_road.id}->{conn.connecting_road.id}\n"  # noqa: E501

        # Verbalize static objects
        if kwargs.get("add_buildings", True):
            if not isinstance(env.unwrapped.simulation.scenario_map, gofi.OMap):
                logger.warning(
                    "Building descriptions are only available for gofi maps.",
                )
            else:
                ret += "\n"
                ret += "- Static objects:\n"
                for obj in scenario_map.objects:
                    if not obj.object_type:
                        continue
                    obj: gofi.StaticObject
                    ret += f"  - {obj.object_type.title()}:\n"
                    ret += f"    - Center: {obj.center}\n"
                    ret += f"    - Boudnary: {util.ndarray2str(obj.boundary_coords[:-1])}\n"  # noqa: E501

        if ret[-1] == "\n":
            ret = ret[:-1]
        return ret

    @staticmethod
    def _add_verbalized_roads(
        ret: str,
        scenario_map: ip.Map,
        **kwargs: dict[str, Any],
    ) -> None:
        for rid, road in scenario_map.roads.items():
            if not road.drivable:
                continue

            ret += f"  - Road {rid}:\n"
            ret += f"    - Length: {road.length} m\n"

            midline = ramer_douglas(
                np.array(road.midline.coords),
                dist=kwargs.get("resolution", 0.02),
            )
            midline = util.ndarray2str(midline)
            ret += f"    - Midline coordinates: {midline}\n"

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
                    ret += "    - Left lanes:\n"
                    for lane in left_lanes:
                        ret += f"      - Lane {rid}:{lane.id}.\n"
                if right_lanes:
                    ret += "    - Right lanes:\n"
                    for lane in right_lanes:
                        ret += f"      - Lane {rid}:{lane.id}.\n"
        return ret

    @staticmethod
    def _verbalize_control_signal(
        signal: str,
        precision: int,
        trajectory: ip.StateTrajectory,
    ) -> tuple[str, str]:
        name = {
            "times": "Time",
            "timesteps": "Timesteps",
            "maneuver": "Maneuvers",
            "macro": "Macro actions",
            "path": "Position",
            "velocity": "Speed",
            "acceleration": "Acceleration",
            "heading": "Heading",
            "angular_velocity": "Steering",
        }[signal]
        data = None

        if signal == "timesteps":
            timesteps = np.array([s.time for s in trajectory.states])
            data = util.ndarray2str(timesteps)
        elif signal == "maneuver":
            data = [s.maneuver for s in trajectory.states]
        elif signal == "macro":
            data = [s.macro_action for s in trajectory.states]
        elif hasattr(trajectory, signal):
            data = util.ndarray2str(getattr(trajectory, signal), precision)
        else:
            error_msg = f"Unknown control signal: {signal}"
            raise ValueError(error_msg)

        return name, data

    @staticmethod
    def convert_query(query: axs.Query) -> tuple[str, str]:
        """Convert the query to query and type descriptions.

        Args:
            query (axs.Query): The query to convert.

        Returns:
            tuple: The query and its type descriptions.

        """
        q_desc = query.query_descriptions()
        q_type_desc = query.query_type_descriptions()

        query_str = ""
        for query_name, syntax in query.queries().items():
            query_str += f"- '{syntax}': {q_desc[query_name]}\n"
        query_types_str = "\n".join([f"- {k}: {v}" for k, v in q_type_desc.items()])
        return query_str[:-1], query_types_str
