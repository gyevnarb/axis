"""Macro action wrapper for IGP2 agent."""

import logging
from copy import copy
from typing import Any, ClassVar

import igp2 as ip
import numpy as np
from numpy import ma

import axs
from envs.axs_igp2 import util

logger = logging.getLogger(__name__)


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
        "FollowLane",
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
        self.scenario_map = scenario_map

    def __repr__(self) -> str:
        """Create representation of the macro action. Used in verbalization."""
        if self.action_segments:
            return (
                f"{self.macro_name}[{self.start_t}-{self.end_t}]"
                # f"({len(self.action_segments)} segments)"
            )
        return f"{self.macro_name}"

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
            groups = cls._group_actions(
                action_segmentations,
                agent_id,
                config,
                scenario_map,
            )

            # Sometimes a single length macro action may appear at the start
            # due to the way the simulation is set up. We remove that here.
            if len(groups) > 1 and groups[0].start_t == groups[0].end_t:
                groups.pop(0)
            ret[agent_id] = groups
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
        return info[self.agent_id].time >= self.start_t

    def done(
        self,
        observation: np.ndarray,
        info: dict[str, ip.AgentState] | None = None,
    ) -> bool:
        """Check whether the macro action is done in the given observation.

        Args:
            observation (Any): The observation to check if the macro action is done.
            info (dict[str, Any] | None): Optional environment info dict.

        """
        if not self.action_segments:
            return True
        macro_action = self.action_segments[0]
        return macro_action.done(ip.Observation(info, self.scenario_map))

    def from_observation(
        self,
        observation: np.ndarray,
        info: dict[str, ip.AgentState] | None = None,
        **kwargs: dict[str, Any],
    ) -> "IGP2MacroAction":
        """Use IGP2's built-in macro actions to set up action segments if possible.

        For the SlowDown, Accelerate, and Stop macro actions, we can directly initialize
        the action segments based on the agent's velocity and acceleration.

        Args:
            observation (np.ndarray): The observation to create the macro action.
            info (dict[str, ip.AgentState] | None): Optional info dict.
            kwargs (dict[str, Any]): Additional optional keyword arguments.
                - fps: The frames per second of the simulation.
                - stop_len: The length of the stop action.

        """
        if not self._ip_macro_applicable(observation, info, **kwargs):
            error_msg = f"{self} is not applicable."
            raise axs.SimulationError(error_msg)

        ma = self.macro_name

        if ma == "Stop":
            ip_macro = ip.StopMA
        elif ma == "ChangeLaneLeft":
            ip_macro = ip.ChangeLaneLeft
        elif ma == "ChangeLaneRight":
            ip_macro = ip.ChangeLaneRight
        elif ma in ["TurnLeft", "TurnRight", "GoStraightJunction", "GiveWay"]:
            ip_macro = ip.Exit
        elif ma == "FollowLane":
            if not ip.Continue.applicable(
                info[self.agent_id],
                self.scenario_map,
            ):
                ip_macro = ip.Exit
            else:
                ip_macro = ip.Continue

        macro = self._get_ip_macro(ip_macro, info, **kwargs)
        self.action_segments = [macro]
        return self

    def next_action(
        self,
        observation: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> ip.Action | None:
        """Return the next action of the macro action."""
        current_macro = self.action_segments[0]
        action = current_macro.next_action(ip.Observation(info, self.scenario_map))
        return {
            self.agent_id: (action, current_macro, current_macro.current_maneuver),
        }

    @property
    def start_t(self) -> int:
        """The start time of the macro action."""
        if isinstance(self.action_segments[0], ip.MacroAction):
            macro_action = self.action_segments[0]
            return macro_action.start_frame[self.agent_id].time
        return super().start_t

    @property
    def end_t(self) -> int | None:
        """The end time of the macro action."""
        if isinstance(self.action_segments[0], ip.MacroAction):
            return None
        return super().end_t

    def _ip_macro_applicable(
        self,
        observation: Any,
        info: dict[str, ip.AgentState] | None = None,
        **kwargs: dict[str, Any],
    ) -> bool:
        """Check whether the IGP2 macro action is applicable in the given state.

        Args:
            observation (Any): The observation to check applicability.
            info (Any | None): Optional environment info dict.
            kwargs (dict[str, Any]): Additional optional keyword arguments.

        """
        state = (info[self.agent_id], self.scenario_map)
        ret = False
        if self.macro_name == "Stop":
            ret = ip.StopMA.applicable(*state)
        elif self.macro_name == "ChangeLaneLeft":
            ret = ip.ChangeLaneLeft.applicable(*state)
        elif self.macro_name == "ChangeLaneRight":
            ret = ip.ChangeLaneRight.applicable(*state)
        elif self.macro_name in ["TurnLeft", "TurnRight", "GoStraightJunction"]:
            ret = ip.Turn.applicable(*state)
        elif self.macro_name == "GiveWay":
            ret = ip.GiveWay.applicable(*state)
        elif self.macro_name == "FollowLane":
            ret = ip.FollowLane.applicable(*state)
        return ret

    def _get_ip_macro(
        self,
        ip_macro: ip.MacroAction,
        info: dict[int, ip.AgentState],
        **kwargs: dict[str, Any],
    ) -> ip.MacroAction:
        """Get the IGP2 macro action object based on the macro name.

        Args:
            ip_macro (ip.MacroAction): The IGP2 macro action class to instantiate.
            info (dict[int, ip.AgentState]): The environment info dict.
            kwargs (dict[str, Any]): Additional optional keyword arguments.

        """
        fps = kwargs["fps"]
        agent_state = info[self.agent_id]
        goal = None
        macro = None

        # Handle stopping without a goal over a set distance
        if self.macro_name == "Stop":
            stop_len = kwargs.get("stop_len", min(agent_state.speed * 2, 20))
            current_lane = self.scenario_map.best_road_at(
                agent_state.position,
                agent_state.heading,
            )
            lane_len = current_lane.length
            current_ds = current_lane.distance_at(agent_state.position)
            if lane_len - current_ds < stop_len:
                error_msg = ("Cannot stop at the current position. "
                             "Too close to end of lane.")
                raise axs.SimulationError(error_msg)
            stop_position = current_lane.point_at(current_ds + stop_len)
            goal = ip.StoppingGoal(stop_position, stop_len)

        ip_ma_args = ip_macro.get_possible_args(agent_state, self.scenario_map, goal)
        for config_dict in ip_ma_args:
            if (
                agent_state.maneuver is not None
                and "GiveWay" in agent_state.maneuver
                and self.macro_name != "GiveWay"
            ):
                config_dict["stop"] = False  # If GiveWay is being overriden, don't stop
            config_dict["open_loop"] = False
            config_dict["fps"] = fps
            _macro = ip_macro(
                ip.MacroActionConfig(config_dict),
                agent_id=self.agent_id,
                frame=info,
                scenario_map=self.scenario_map,
            )
            if ip_macro == ip.Exit:
                direction = kwargs.get("turn_direction")
                if direction is None:
                    direction = self.get_turn_direction()
                if direction == _macro.orientation:
                    macro = _macro
                    break
            else:
                macro = _macro

        if macro is None:
            macro_name = self.macro_name
            if "turn_direction" in kwargs:
                macro_name = {
                    1: "TurnLeft",
                    -1: "TurnRight",
                    0: "GoStraightJunction",
                }[kwargs["turn_direction"]]
            error_msg = f"Macro {macro_name} is not applicable."
            raise axs.SimulationError(error_msg)
        return macro

    def get_turn_direction(self) -> int:
        """Get the turn direction based on the macro name.

        Args:
            macro_name (str): The name of the macro action.

        Returns:
            int: The turn direction (-1 for left, 1 for right, 0 for straight).

        """
        if self.macro_name not in ["TurnLeft", "TurnRight", "GoStraightJunction"]:
            logger.warning("Macro %s is not a standard turn direction. "
                           "Defaulting to going straight", self.macro_name)

        if self.macro_name == "TurnLeft":
            return 1
        if self.macro_name == "TurnRight":
            return -1
        return 0

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
            action_sequences[slicer] = [
                (actions[:-1] + (turn_type,), a)
                for actions, a in action_sequences[slicer]
            ]

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
        raw_action = np.array(
            [trajectory.acceleration[inx], trajectory.angular_velocity[inx]],
        )

        if trajectory.acceleration[inx] < -eps:
            action_names.append("SlowDown")
        elif trajectory.acceleration[inx] > eps:
            action_names.append("Accelerate")

        if trajectory.velocity[inx] <= trajectory.VELOCITY_STOP:
            action_names.append("Stop")

        if len(action_names) > 0 and action_names[-1] == "Stop":
            return tuple(action_names), raw_action

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
            elif "FollowLane" in state.maneuver or (
                "StopCL" in state.maneuver
                and trajectory.velocity[inx] > trajectory.VELOCITY_STOP
            ):
                action_names.append("FollowLane")

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
