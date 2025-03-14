""" This module contains the main agent class for the AXS framework. """
import logging
from typing import Any, Dict

import gymnasium as gym
from vllm import LLM

from axs.config import Config
from axs.memory import SemanticMemory, EpisodicMemory
from axs.simulator import Simulator
from axs.verbalize import Verbalizer


logger = logging.getLogger(__name__)


class AXSAgent:
    """ The main agent class for the AXS framework. """

    def __init__(self,
                 config: Config,
                 simulator_env: gym.Env = None):
        """ Initialize the AXS agent with the parameters.

        Args:
            model (str): The LLM model name to be loaded through vLLM.
            initial_observation (Any): The initial observation from the environment.
            initial_info (Dict[str, Any]): The initial information dict from the environment.
            simulator_env (gym.Env): Optional environment to be used for simulation.
                                     If not given, a new internal environment will be created.
        """
        self.config = config

        # Memory components
        self._semantic_memory = SemanticMemory({
                "observations": [],
                "actions": [],
                "infos": []
            })
        self._episodic_memory = EpisodicMemory()

        # Procedural components
        self._simulator = Simulator(config.env, simulator_env)  # Internal simulator
        self._llm = LLM(config.llm.model, **config.llm.model_kwargs)

        # Utility components
        self._macro_action = 
        self._verbalizer = Verbalizer()
        self._explanation_prompt = Prompt()

    def explain(self, user_prompt: str) -> str:
        """ Explain behaviour based on the user's prompt. """
        messages = [
            {"role": "system", "content": "TODO: Add system message."}
        ]

        observations = self._semantic_memory.retrieve("observations")
        actions = self._semantic_memory.retrieve("actions")
        macro_actions = MacroAction.wrap(actions)

        txt_obs = verbalizer.convert(observations)
        txt_acts = verbalizer.convert(macro_actions)

        query_prompt = prompt.query_prompt(user=user_prompt,
                                           states=txt_obs,
                                           actions=txt_acts)
        messages.append(query_prompt)

        n = 0
        explanation, prev_explanation = None, None
        sampling_params = SamplingParams(**sampling_params)
        while n < n_max and distance(explanation, prev_explanation) > delta:
            # Simulator interrogation
            query_output = llm.chat(messages)
            simulation_query = query_output.outputs[0].content
            sim_states, sim_actions, rewards = simulator.query(simulation_query)
            episodic_memory.learn({"states": sim_states, "actions": sim_actions, "rewards": rewards})

            # Explanation synthesis
            txt_sim_states = verbalizer.convert(sim_states)
            txt_sim_actions = verbalizer.convert(sim_actions)
            txt_rewards = verbalizer.convert(rewards)
            explanation_prompt = prompt.explanation_prompt(states=txt_sim_states,
                                                        actions=txt_sim_actions,
                                                        rewards=txt_rewards)
            messages.append(explanation_prompt)
            explanation_output = llm.chat(messages)
            explanation = explanation_output.outputs[0].content

            prev_explanation = explanation
            n += 1

    def reset(self) -> None:
            """ Reset the agent. """
            self._semantic_memory.reset()
            self._episodic_memory.reset()


    @property
    def semantic_memory(self) -> SemanticMemory:
        """ The agent's semantic memory. """
        return self._semantic_memory

    @property
    def episodic_memory(self) -> EpisodicMemory:
        """ The agent's episodic memory. """
        return self._episodic_memory

    @property
    def llm(self) -> LLM:
        """ The agent's LLM model. """
        return self._llm

    @property
    def simulator(self) -> Simulator:
        """ The agent's simulator. """
        return self._simulator

    @property
    def verbalizer(self):
        return self._verbalizer
