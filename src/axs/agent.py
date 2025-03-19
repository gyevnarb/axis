"""Contains the main agent class for the AXS framework."""

import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

from axs import SupportedEnv
from axs.config import Config
from axs.llm import LLMWrapper
from axs.macroaction import MacroAction
from axs.memory import EpisodicMemory, SemanticMemory
from axs.prompt import Prompt
from axs.simulator import Simulator
from axs.verbalize import Verbalizer

logger = logging.getLogger(__name__)


class AXSAgent:
    """The main agent class for the AXS framework.

    The agent may be used with both offline inference through vLLM
    or online inference through the OpenAI API.
    """

    def __init__(
        self,
        config: Config,
        simulator_env: SupportedEnv | None = None,
    ) -> "AXSAgent":
        """Initialize the AXS agent with the parameters.

        Args:
            config (Config): The configuration object for the agent.
            simulator_env (SupportedEnv): Optional environment to be used for simulation.
                            If not given, a new internal environment will be created.

        """
        self.config = config

        # Prompting components
        self._system_prompt = Prompt(config.axs.system_template)
        self._query_prompt = Prompt(config.axs.query_template)
        self._explanation_prompt = Prompt(config.axs.explanation_template)

        # Memory components
        self._semantic_memory = SemanticMemory(
            {"observations": [], "actions": [], "infos": []},
        )
        self._episodic_memory = EpisodicMemory()

        # Procedural components
        self._simulator = Simulator(config.env, simulator_env)  # Internal simulator
        self._llm = LLMWrapper(config.llm)

        # Utility components
        self._macro_action = MacroAction.get(config.axs.macro_action.name)
        self._verbalizer = Verbalizer.get(config.axs.verbalizer.name)

    def explain(self, user_prompt: str) -> str:
        """Explain behaviour based on the user's prompt.

        Args:
            user_prompt (str): The user's prompt to the agent.

        """
        logger.info("Explaining behaviour based on user prompt: %s", user_prompt)

        messages = [
            {
                "role": "developer",
                "content": self._system_prompt.fill(n_max=self.config.axs.n_max),
            },
        ]

        observations = self._semantic_memory.retrieve("observations")
        actions = self._semantic_memory.retrieve("actions")
        infos = self._semantic_memory.retrieve("infos")

        macro_actions = self._macro_action.wrap(
            self.config.axs.macro_action,
            actions,
            observations,
            infos,
        )
        if not isinstance(macro_actions, dict) and not all(
            isinstance(k, int) for k in macro_actions
        ):
            error_msg = (f"Macro actions must be a dictionary with "
                         f"int agent ids as keys. Got: {macro_actions}")
            raise ValueError(error_msg)

        context = self._verbalizer.convert(
            self._simulator.env.unwrapped,
            observations,
            macro_actions,
            infos,
            self.config.verbalizer.params,
        )

        query_prompt = self._query_prompt.fill(
            question=user_prompt,
            macro_names=self._macro_action.macro_names,
            context=context,
        )
        messages.append(query_prompt)

        n = 0
        explanation, prev_explanation = None, None
        while (
            n < self.config.axs.n_max
            and distance(explanation, prev_explanation) > self.config.axs.delta
        ):
            # Simulator interrogation
            query_output = self._llm.chat(messages)
            simulation_query = query_output.outputs[0].content
            self._simulator.set_state(start_state)
            sim_states, sim_actions, rewards = self._simulator.query(simulation_query)
            self._episodic_memory.learn(
                {"states": sim_states, "actions": sim_actions, "rewards": rewards},
            )

            # Explanation synthesis
            txt_sim_states = self._verbalizer.convert(sim_states)
            txt_sim_actions = self._verbalizer.convert(sim_actions)
            txt_rewards = self._verbalizer.convert(rewards)
            explanation_prompt = self._explanation_prompt.fill(
                states=txt_sim_states,
                actions=txt_sim_actions,
                rewards=txt_rewards,
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

    def save_state(self, path: str) -> None:
        """Save the agent's state to a file except the LLM."""
        statedict = {
            "semantic_memory": self._semantic_memory,
            "episodic_memory": self._episodic_memory,
            "simulator": self._simulator,
            "verbalizer": self._verbalizer,
            "macro_action": self._macro_action,
        }

        with Path.open(path, "wb") as f:
            pickle.dump(statedict, f)

    def load_state(self, path: str) -> None:
        """Load the agent's state from a file except for the LLM."""
        with Path.open(path, "rb") as f:
            statedict = pickle.load(f)
        for key, value in statedict.items():
            setattr(self, "_" + key, value)

    @property
    def semantic_memory(self) -> SemanticMemory:
        """The agent's semantic memory."""
        return self._semantic_memory

    @property
    def episodic_memory(self) -> EpisodicMemory:
        """The agent's episodic memory."""
        return self._episodic_memory

    @property
    def llm(self) -> LLMWrapper:
        """The agent's LLM model."""
        return self._llm

    @property
    def simulator(self) -> Simulator:
        """The agent's simulator."""
        return self._simulator

    @property
    def verbalizer(self) -> Verbalizer:
        """The agent's verbalizer."""
        return self._verbalizer
