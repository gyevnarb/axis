"""IGP2 policy wrapper for agents."""

from copy import copy

import gofi
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

    def reset(
        self,
        observations: list[np.ndarray],
        infos: list[dict[int, ip.AgentState]],
        env: ip.simplesim.SimulationEnv | None = None,
    ) -> None:
        """Reset the internal state of the policy."""
        self.agent.reset()
        if isinstance(self.agent, gofi.GOFIAgent):
            self.agent._forced_visible_agents = env.simulation.agents[
                self.agent.agent_id
            ]._forced_visible_agents
        if observations or infos:
            self.update(observations, infos)

    def update(
        self,
        observations: list[np.ndarray],
        infos: list[dict[int, ip.AgentState]],
    ) -> None:
        """Update the internal policy state."""
        if not hasattr(self.agent, "observations"):
            return
        self.agent._vehicle = type(self.agent._vehicle)(
            infos[-1][self.agent.agent_id],
            self.agent.metadata,
            self.agent.fps,
        )
        trajectories = util.infos2traj(infos, None, self.agent.fps)
        self.agent._trajectory_cl.extend(trajectories[self.agent.agent_id])
        for agent_id, trajectory in trajectories.items():
            if agent_id not in infos[-1]:
                if agent_id in self.agent.observations:
                    self.agent.observations.pop(agent_id)
                continue
            self.agent.observations[agent_id] = (copy(trajectory), copy(infos[0]))

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
        action = self.agent.next_action(ip_observation)
        return {
            self.agent.agent_id: (
                action,
                self.agent.current_macro,
                self.agent.current_macro.current_maneuver,
            ),
        }

    @classmethod
    def create(
        cls,
        env: axs.SupportedEnv | None = None,
    ) -> dict[int, "IGP2Policy"]:
        """Create a policy for each agent in the environment.

        Args:
            env (SupportedEnv): The environment to create policies for.

        """
        if env is None:
            error_msg = "Environment must be provided."
            raise ValueError(error_msg)
        return {
            aid: cls(agent, env.unwrapped.scenario_map)
            for aid, agent in env.unwrapped.simulation.agents.items()
        }
