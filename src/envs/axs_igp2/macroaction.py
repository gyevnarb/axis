"""Macro action wrapper for IGP2 agent."""

from copy import copy
from typing import Any, ClassVar

import igp2 as ip
import numpy as np
from numpy import ma

import axs
from envs.axs_igp2 import util


class IGP2MacroAction(axs.MacroAction):
    """Macro action wrapper for IGP2 agent.

    The wrapper takes the agent state information and converts into text.
    """

    macro_names: ClassVar[list[str]] = [
        # "SlowDown",
        # "Accelerate",
        "Stop",
        "ChangeLaneLeft",
        "ChangeLaneRight",
        "TurnLeft",
        "TurnRight",
        "GoStraightJunction",
        "GiveWay",
        "GoStraight",
    ]

    def __init__(
        self,
        macro_name: str,
        agent_id: int | None = None,
        config: axs.MacroActionConfig | None = None,
        action_segments: list[axs.ActionSegment] | None = None,
        scenario_map: ip.Map | None = None,
    ) -> "IGP2MacroAction":
        """Initialize the macro action."""
        super().__init__(macro_name, agent_id, config, action_segments)
        self._scenario_map = scenario_map

    def __repr__(self) -> str:
        """Create representation of the macro action. Used in verbalization."""
        if self.action_segments:
            return (
                f"{self.macro_name}[{self.start_t}-{self.end_t}]"
                # f"({len(self.action_segments)} segments)"
            )
        return f"{self.macro_name}[empty]"

    def __str__(self) -> str:
        """Create string representation of the macro action."""
        return self.__repr__()

    @classmethod
    def wrap(
        cls,
        config: axs.MacroActionConfig,
        actions: list[np.ndarray],  # noqa: ARG003
        observations: list[np.ndarray] | None = None,  # noqa: ARG003
        infos: list[str, dict[Any]] | None = None,
        env: ip.simplesim.SimulationEnv | None = None,
    ) -> dict[int, list["IGP2MacroAction"]]:
        """Segment a trajectory into different actions and sorted with time.

        Also stores results in place and overrides previously stored actions.

        Args:
            config (MacroActionConfig): The configuration for the macro action.
            actions (List[np.ndarray]): An agent trajectory to
                    segment into macro actions.
            observations (List[Dict[str, np.ndarray]]): The environment
                    observation sequence.
            infos (List[Dict[int, ip.AgentState]]): Optional list of
                    agent states from the environment.
            env (SupportedEnv): The environment to extract the agent states.

        """
        scenario_map = env.scenario_map

        trajectories = util.infos2traj(infos, fps=env.fps)

        ret = {}
        for agent_id, trajectory in trajectories.items():
            cls._fix_initial_state(trajectory)
            action_sequences = []
            for inx in range(len(trajectory.times)):
                matched_actions = cls._match_actions(
                    config.params["eps"],
                    scenario_map,
                    trajectory,
                    inx,
                )
                action_sequences.append(matched_actions)
            action_segmentations = cls._segment_actions(
                config,
                trajectory,
                action_sequences,
            )
            ret[agent_id] = cls._group_actions(
                action_segmentations,
                agent_id,
                config,
                scenario_map,
            )
        return ret

    def applicable(
        self,
        observation: Any,
        info: dict[str, ip.AgentState] | None = None,
    ) -> bool:
        """Check if the macro action is applicable in the given observation.

        Args:
            observation (Any): The observation to check applicability.
            info (Any | None): Optional environment info dict.

        """
        state = (info[self.agent_id], self.scenario_map)
        ret = False
        if self.name == "Stop":
            ret = ip.StopMA.applicable(*state)
        elif self.name == "ChangeLaneLeft":
            ret = ip.ChangeLaneLeft.applicable(*state)
        elif self.name == "ChangeLaneRight":
            ret = ip.ChangeLaneRight.applicable(*state)
        elif self.name in ["TurnLeft", "TurnRight", "GoStraightJunction"]:
            ret = ip.Turn.applicable(*state)
        elif self.name == "GiveWay":
            ret = ip.GiveWay.applicable(*state)
        elif self.name == "GoStraight":
            ret = ip.FollowLane.applicable(*state)
        return ret

    def from_observation(
        self,
        observation: np.ndarray,
        info: dict[str, ip.AgentState] | None = None,
    ) -> None:
        """Use IGP2's built-in macro actions to set up action segments if possible.

        For the SlowDown, Accelerate, and Stop macro actions, we can directly initialize
        the action segments based on the agent's velocity and acceleration.

        Args:
            observation (np.ndarray): The observation to create the macro action.
            info (dict[str, ip.AgentState] | None): Optional info dict.

        """
        if self.name == "Stop":
            ip_macro = ip.StopMA
        elif self.name == "ChangeLaneLeft":
            ip_macro = ip.ChangeLaneLeft
        elif self.name == "ChangeLaneRight":
            ip_macro = ip.ChangeLaneRight
        elif self.name in ["TurnLeft", "TurnRight", "GoStraightJunction", "GiveWay"]:
            ip_macro = ip.Exit
        elif self.name == "GoStraight":
            ip_macro = ip.Continue

        agent_state = info[self.agent_id]
        ip_agent = ip.MacroAgent(self.agent_id, agent_state)
        ip_observation = ip.Observation(info, self.scenario_map)
        ip_ma_args = ip_macro.get_possible_args(agent_state, self.scenario_map)
        for config_dict in ip_ma_args:
            if "GiveWay" in agent_state.maneuver and self.name != "GiveWay":
                config_dict["stop"] = False  # If GiveWay is being overriden, don't stop
            macro = ip_agent.update_macro_action(ip_macro, config_dict, ip_observation)
            if ip_macro == ip.Exit:
                direction = (
                    1
                    if self.name == "TurnLeft"
                    else -1
                    if self.name == "TurnRight"
                    else 0
                )
                if direction == macro.direction:
                    break
        self.action_segments = []
        self._agent = ip_agent

    def next_action(
        self,
        observation: Any | None = None,
        info: dict[str, Any] | None = None,
        env: ip.simplesim.SimulationEnv | None = None,
    ) -> ip.Action:
        """Return the next action of the macro action."""
        return self._agent.next_action(ip.Observation(info, env.scenario_map))

    @classmethod
    def _group_actions(
        cls,
        action_segmentations: list[axs.ActionSegment],
        agent_id: int,
        config: axs.MacroActionConfig,
        scenario_map: ip.Map,
    ) -> list["IGP2MacroAction"]:
        """Group action segments to macro actions by IGP2 maneuver."""
        ret, group = [], []
        prev_man = action_segmentations[0].name[-1]
        for segment in action_segmentations:
            man = segment.name[-1]
            if prev_man != man:
                ret.append(
                    cls(prev_man, agent_id, config, group, scenario_map=scenario_map),
                )
                group = []
            group.append(segment)
            prev_man = man
        ret.append(cls(prev_man, agent_id, config, group, scenario_map=scenario_map))
        return ret

    @staticmethod
    def _segment_actions(
        config: axs.MacroActionConfig,
        trajectory: ip.StateTrajectory,
        action_sequences: list[tuple[tuple[str, ...], np.ndarray]],
    ) -> list[axs.ActionSegment]:
        """Group the action sequences into segments, potentially fixing turning actions.

        Args:
            config (MacroActionConfig): The configuration for the macro action.
            trajectory (ip.StateTrajectory): The trajectory of the agent.
            action_sequences (List[Tuple[Tuple[str, ...], np.ndarray]]):
                    The action sequences of the agent.

        """
        eps = config.params["eps"]
        idx = [
            a[-1] in ["TurnLeft", "TurnRight", "GoStraightJunction"]
            for a, _ in action_sequences
        ]
        avels = ma.array(trajectory.angular_velocity, mask=idx)
        for slicer in ma.clump_masked(avels):
            mean_avel = trajectory.angular_velocity[slicer].mean()
            if mean_avel > eps / 2:
                turn_type = "TurnLeft"
            elif mean_avel < -eps / 2:
                turn_type = "TurnRight"
            else:
                turn_type = "GoStraightJunction"
            for actions, _ in action_sequences[slicer]:
                actions[-1] = turn_type

        # aggregate same actions during a period
        action_segmentations = []
        actions, times = [], []
        action_names, previous_action_names = None, None
        start_time = int(trajectory.states[0].time)
        for inx, (action_names, action) in enumerate(action_sequences, start_time):
            if (
                previous_action_names is not None
                and previous_action_names != action_names
            ):
                action_segmentations.append(
                    axs.ActionSegment(times, actions, previous_action_names),
                )
                actions, times = [], []
            times.append(inx)
            actions.append(action)
            previous_action_names = action_names
        action_segmentations.append(axs.ActionSegment(times, actions, action_names))

        return action_segmentations

    @staticmethod
    def _match_actions(
        eps: float,
        scenario_map: ip.Map,
        trajectory: ip.StateTrajectory,
        inx: int,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        """Segment the trajectory into different actions and sorted with time.

        Args:
            eps (float): The epsilon value for comparison.
            scenario_map (ip.Map): The road layout of the scenario.
            trajectory (ip.StateTrajectory): Trajectory to segment into macro actions.
            inx (int): The index of the trajectory to segment.

        Returns:
            Tuple[str, ...]: A list of macro action names for the agent at given index.
            np.ndarray: Raw action (acceleration-steering) of the agent at given index.

        """
        action_names = []
        state = trajectory.states[inx]

        if trajectory.acceleration[inx] < -eps:
            action_names.append("SlowDown")
        elif trajectory.acceleration[inx] > eps:
            action_names.append("Accelerate")

        if trajectory.velocity[inx] < trajectory.VELOCITY_STOP:
            action_names.append("Stop")

        if state.macro_action is not None:
            if "ChangeLaneLeft" in state.macro_action:
                action_names.append("ChangeLaneLeft")
            elif "ChangeLaneRight" in state.macro_action:
                action_names.append("ChangeLaneRight")

        if state.maneuver is not None:
            if "Turn" in state.maneuver:
                road_in_roundabout = None
                if scenario_map is not None:
                    road = scenario_map.best_road_at(state.position, state.heading)
                    road_in_roundabout = scenario_map.road_in_roundabout(road)
                angular_vel = trajectory.angular_velocity[inx]
                if road_in_roundabout:
                    action_names.append("GoStraightJunction")
                elif angular_vel > eps:
                    action_names.append("TurnLeft")
                elif angular_vel < -eps:
                    action_names.append("TurnRight")
                else:
                    action_names.append("GoStraightJunction")
            elif "GiveWay" in state.maneuver:
                action_names.append("GiveWay")
            elif "FollowLane" in state.maneuver:
                action_names.append("GoStraight")
        raw_action = np.array(
            [trajectory.acceleration[inx], trajectory.angular_velocity[inx]],
        )
        return tuple(action_names), raw_action

    @staticmethod
    def _fix_initial_state(trajectory: ip.StateTrajectory) -> None:
        """Fix missing initial maneuver and macro name.

        The initial frame is often missing macro and maneuver information
        due to the planning flow of IGP2. This function fills in the missing
        information using the second state.

        Args:
            trajectory: The StateTrajectory whose first state is missing macro action or
                maneuver information.

        """
        if (
            len(trajectory.states) > 1
            and trajectory.states[0].time == 0
            and trajectory.states[0].macro_action is None
            and trajectory.states[0].maneuver is None
        ):
            trajectory.states[0].macro_action = copy(trajectory.states[1].macro_action)
            trajectory.states[0].maneuver = copy(trajectory.states[1].maneuver)
