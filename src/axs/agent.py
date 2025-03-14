"""This module contains the main agent class for the AXS framework."""

import logging
from typing import Any, Dict

import gymnasium as gym
from vllm import LLM
from vllm import SamplingParams

from axs.config import Config
from axs.prompt import Prompt
from axs.macro_action import MacroActionFactory
from axs.memory import SemanticMemory, EpisodicMemory
from axs.simulator import Simulator
from axs.verbalize import Verbalizer


logger = logging.getLogger(__name__)


class AXSAgent:
    """The main agent class for the AXS framework."""

    def __init__(self, config: Config, simulator_env: gym.Env = None):
        """Initialize the AXS agent with the parameters.

        Args:
            model (str): The LLM model name to be loaded through vLLM.
            initial_observation (Any): The initial observation from the environment.
            initial_info (Dict[str, Any]): The initial information dict from the environment.
            simulator_env (gym.Env): Optional environment to be used for simulation.
                                     If not given, a new internal environment will be created.
        """
        self.config = config
        self.n_max = config.axs.n_max
        self.delta = config.axs.delta

        # Memory components
        self._semantic_memory = SemanticMemory(
            {"observations": [], "actions": [], "infos": []}
        )
        self._episodic_memory = EpisodicMemory()

        # Procedural components
        self._simulator = Simulator(config.env, simulator_env)  # Internal simulator
        self._llm = LLM(config.llm.model, **config.llm.model_kwargs)
        self._sampling_params = SamplingParams(**config.llm.sampling_params)

        # Utility components
        self._macro_action = MacroActionFactory.create(config)
        self._verbalizer = Verbalizer()
        self._query_prompt = Prompt(config.axs.query_template)
        self._explanation_prompt = Prompt(config.axs.explanation_template)

    def explain(self, user_prompt: str) -> str:
        """Explain behaviour based on the user's prompt."""
        messages = [{"role": "system", "content": self.config.axs.system_prompt}]

        observations = self._semantic_memory.retrieve("observations")
        actions = self._semantic_memory.retrieve("actions")
        macro_actions = self._macro_action.wrap(actions)

        txt_obs = self._verbalizer.convert(observations)
        txt_acts = self._verbalizer.convert(macro_actions)

        query_prompt = self._query_prompt.fill(
            user=user_prompt, states=txt_obs, actions=txt_acts
        )
        messages.append(query_prompt)

        n = 0
        explanation, prev_explanation = None, None
        while (
            n < self.n_max
            and distance(explanation, prev_explanation) > self.delta
        ):
            # Simulator interrogation
            query_output = self._llm.chat(messages)
            simulation_query = query_output.outputs[0].content
            sim_states, sim_actions, rewards = self._simulator.query(simulation_query)
            self._episodic_memory.learn(
                {"states": sim_states, "actions": sim_actions, "rewards": rewards}
            )

            # Explanation synthesis
            txt_sim_states = self._verbalizer.convert(sim_states)
            txt_sim_actions = self._verbalizer.convert(sim_actions)
            txt_rewards = self._verbalizer.convert(rewards)
            explanation_prompt = self._explanation_prompt.fill(
                states=txt_sim_states, actions=txt_sim_actions, rewards=txt_rewards
            )
            messages.append(explanation_prompt)
            explanation_output = self._llm.chat(messages)
            explanation = explanation_output.outputs[0].content

            prev_explanation = explanation
            n += 1

    def reset(self) -> None:
        """Reset the agent."""
        self._semantic_memory.reset()
        self._episodic_memory.reset()

    @property
    def semantic_memory(self) -> SemanticMemory:
        """The agent's semantic memory."""
        return self._semantic_memory

    @property
    def episodic_memory(self) -> EpisodicMemory:
        """The agent's episodic memory."""
        return self._episodic_memory

    @property
    def llm(self) -> LLM:
        """The agent's LLM model."""
        return self._llm

    @property
    def simulator(self) -> Simulator:
        """The agent's simulator."""
        return self._simulator

    @property
    def verbalizer(self):
        """ The agent's verbalizer. """
        return self._verbalizer
