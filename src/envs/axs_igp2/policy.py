"""IGP2 policy wrapper for agents."""

from copy import deepcopy

import igp2 as ip
import numpy as np

import axs


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

    def next_action(
        self, observation: list[np.ndarray], info: dict[int, ip.AgentState],
    ) -> ip.Action:
        """Get the next action from the policy.

        Args:
            observation (Any): The observation from the environment.
            info (dict[str, Any] | None): The info dict from the environment.

        """
        ip_observation = ip.Observation(info, self.scenario_map)
        return self.agent.next_action(ip_observation)
