"""IGP2 policy wrapper for agents."""

from copy import copy

import igp2 as ip
import numpy as np

import axs
from envs.axs_igp2 import util


class IGP2Policy(axs.Policy):
    """IGP2 policy wrapper for agents."""

    def __init__(self, agent: ip.Agent, scenario_map: ip.Map) -> "IGP2Policy":
        """Initialize the IGP2 policy with the agent.

        Args:
            agent (ip.Agent): The IGP2 agent to wrap.
            scenario_map (ip.Map): The scenario map for the agent.

        """
        self.agent = agent
        self.scenario_map = scenario_map

    def update(
        self, observations: list[np.ndarray], infos: list[dict[int, ip.AgentState]],
    ) -> None:
        """Update the internal policy state."""
        self.agent._initial_state = infos[-1][self.agent.agent_id]
        self.agent.reset()
        trajectories = util.infos2traj(infos, None, self.agent.fps)
        for agent_id, trajectory in trajectories.items():
            if agent_id not in infos[-1]:
                continue
            self.agent.observations[agent_id] = (trajectory, copy(infos[0]))

    def next_action(
        self,
        observation: list[np.ndarray],
        info: dict[int, ip.AgentState],
    ) -> dict[int, ip.Action]:
        """Get the next action from the policy.

        Args:
            observation (Any): The observation from the environment.
            info (dict[str, Any] | None): The info dict from the environment.

        """
        ip_observation = ip.Observation(info, self.scenario_map)
        self.agent
        return {self.agent.agent_id: self.agent.next_action(ip_observation)}
