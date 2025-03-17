"""Macro action wrapper for IGP2 agent."""

from copy import copy
from typing import List, Dict, Any, Tuple, Generator
from collections import defaultdict

import numpy as np
from numpy import ma
import igp2 as ip

from axs.macroaction.base import MacroAction, ActionSegment
from axs.config import MacroActionConfig


class IGP2MacroAction(MacroAction):
    """Macro action wrapper for IGP2 agent.
    The wrapper takes the agent state information and converts into text.
    """

    macro_names = [
        "SlowDown",
        "Accelerate",
        "Stop",
        "ChangeLaneLeft",
        "ChangeLaneRight",
        "TurnLeft",
        "TurnRight",
        "GoStraightJunction",
        "GiveWay",
        "GoStraight",
    ]

    def __repr__(self) -> str:
        """String representation of the macro action. Used in verbalization."""
        if self.action_segments:
            return (
                f"{self.macro_name}[{self.start_t}-{self.end_t}]"
                f"({len(self.action_segments)} segments)"
            )
        return f"{self.macro_name}[empty]"

    @classmethod
    def wrap(
        cls, config: MacroActionConfig, actions, observations, infos=None
    ) -> Dict[int, List["IGP2MacroAction"]]:
        """Segment the trajectory into different actions and sorted with time.
        Also stores results in place and overrides previously stored actions.

        Args:
            config (MacroActionConfig): The configuration for the macro action
            actions (List[np.ndarray]): An agent trajectory to
                    segment into macro actions.
            observations (List[Dict[str, np.ndarray]]): The environment
                    observation sequence.
            infos (List[Dict[int, ip.AgentState]]): Optional list of
                    agent states from the environment.
        """
        trajectories = defaultdict(list)
        for frame in infos:
            for agent_id, state in frame.items():
                trajectories[agent_id].append(state)

        ret = {}
        for agent_id, states in trajectories.items():
            trajectory = ip.StateTrajectory(None, states)
            cls._fix_initial_state(trajectory)
            action_sequences = []
            for inx in range(len(trajectory.times)):
                matched_actions = cls._match_actions(config, trajectory, inx)
                action_sequences.append(matched_actions)
            action_segmentations = cls._segment_actions(
                config, trajectory, action_sequences
            )
            ret[agent_id] = cls._group_actions(action_segmentations)
        return ret

    @classmethod
    def unwrap(cls, macro_actions: List["MacroAction"]) -> Generator[Any, None, None]:
        """Unwrap the macro actions into low-level actions. Returns a generator."""
        for macro_action in macro_actions:
            for segment in macro_action.action_segments:
                yield from segment.actions

    @classmethod
    def _group_actions(
        cls, action_segmentations: List[ActionSegment]
    ) -> List["IGP2MacroAction"]:
        """Group action segments to macro actions by IGP2 maneuver."""
        ret, group = [], []
        prev_man = action_segmentations[0].name[-1]
        for segment in action_segmentations:
            man = segment.name[-1]
            if prev_man != man:
                ret.append(cls(prev_man, group))
                group = []
            group.append(segment)
            prev_man = man
        ret.append(cls(prev_man, group))
        return ret

    @staticmethod
    def _segment_actions(
        config: MacroActionConfig,
        trajectory: ip.StateTrajectory,
        action_sequences: List[Tuple[Tuple[str, ...], np.ndarray]],
    ) -> List[ActionSegment]:
        """Group the action sequences into segments based,
        potentially fixing erroneous turning actions.

        Args:
            config (MacroActionConfig): The configuration for the macro action.
            trajectory (ip.StateTrajectory): The trajectory of the agent.
            action_sequences (List[Tuple[Tuple[str, ...], np.ndarray]]):
                    The action sequences of the agent.
        """
        # Fix other turning actions appearing due to variable angular velocity.
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
                    ActionSegment(times, actions, previous_action_names)
                )
                actions, times = [], []
            times.append(inx)
            actions.append(action)
            previous_action_names = action_names
        action_segmentations.append(ActionSegment(times, actions, action_names))

        return action_segmentations

    @staticmethod
    def _match_actions(
        config: MacroActionConfig, trajectory: ip.StateTrajectory, inx: int
    ) -> Tuple[Tuple[str, ...], np.ndarray]:
        """Segment the trajectory into different actions and sorted with time.

        Args:
            config (MacroActionConfig): The configuration for the macro action.
            trajectory (ip.StateTrajectory): An agent trajectory to segment into macro actions.
            inx (int): The index of the trajectory to segment.

        Returns:
            Tuple[str, ...]: A list of macro action names for the agent at the given index.
            np.ndarray: The raw action (acceleration and steering) of the agent at the given index.
        """
        eps = config.params["eps"]
        scenario_map = config.params["scenario_map"]
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
            [trajectory.acceleration[inx], trajectory.angular_velocity[inx]]
        )
        return tuple(action_names), raw_action

    @staticmethod
    def _fix_initial_state(trajectory: ip.StateTrajectory):
        """The initial frame is often missing macro and maneuver information
        due to the planning flow of IGP2. This function fills in the missing
        information using the second state.

        Args:
            trajectory: The StateTrajectory whose first state
                        is missing macro action or maneuver information.
        """
        if (
            len(trajectory.states) > 1
            and trajectory.states[0].time == 0
            and trajectory.states[0].macro_action is None
            and trajectory.states[0].maneuver is None
        ):
            trajectory.states[0].macro_action = copy(trajectory.states[1].macro_action)
            trajectory.states[0].maneuver = copy(trajectory.states[1].maneuver)
