"""Contains the main agent class for the AXS framework."""

import logging
import pickle
from pathlib import Path
from typing import Any

from axs.config import Config, SupportedEnv
from axs.llm import LLMWrapper
from axs.macroaction import MacroAction
from axs.memory import EpisodicMemory, SemanticMemory
from axs.policy import Policy
from axs.prompt import Prompt
from axs.query import Query
from axs.simulator import SimulationError, Simulator
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
        agent_policies: dict[int, Policy],
        simulator_env: SupportedEnv | None = None,
        **kwargs: dict[str, Any],
    ) -> "AXSAgent":
        """Initialize the AXS agent with the parameters.

        Args:
            config (Config): The configuration object for the agent.
            agent_policies (dict[int, Policy]): Agent policies used in the simulator.
            simulator_env (SupportedEnv): Optional environment to use for simulation.
                If not given, a new internal environment will be created.
                Note, if an existing environment is passed then
                the environment will be changed in-place by the agent.
            kwargs (dict): Additional optional keyword arguments.
                If not given, the config file will be used.
                - macro_type (type[MacroAction]): The type of macro action to use.
                - verbalizer_type (type[Verbalizer]): The type of verbalizer to use.
                - query_type (type[Query]): The type of query to use.

        """
        if not isinstance(config, Config):
            error_msg = f"Config must be an instance of Config. Got: {config}"
            raise TypeError(error_msg)
        if not isinstance(agent_policies, dict) or not all(
            isinstance(k, int) and isinstance(v, Policy)
            for k, v in agent_policies.items()
        ):
            error_msg = (
                f"Agent policies must be a dictionary with "
                f"int agent ids as keys and Policy as values. Got: {agent_policies}"
            )
            raise ValueError(error_msg)
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
        self._simulator = Simulator(config.env, agent_policies, simulator_env)
        self._llm = LLMWrapper(config.llm)
        self._distance = lambda x, y: True  # TODO: Placeholder

        # Utility components
        if "macro_type" in kwargs:
            macro_type: type[MacroAction] = kwargs["macro_type"]
            if not issubclass(macro_type, MacroAction):
                error_msg = (
                    f"Macro type must be a subclass of MacroAction. Got: {macro_type}"
                )
                raise ValueError(error_msg)
            self._macro_action = macro_type
        else:
            self._macro_action = MacroAction.get(config.axs.macro_action.type_name)

        if "verbalizer_type" in kwargs:
            verbalizer_type: type[Verbalizer] = kwargs["verbalizer_type"]
            if not issubclass(verbalizer_type, Verbalizer):
                error_msg = (
                    f"Verbalizer type must be a subclass of Verbalizer. "
                    f"Got: {verbalizer_type}"
                )
                raise ValueError(error_msg)
            self._verbalizer = verbalizer_type
        else:
            self._verbalizer = Verbalizer.get(config.axs.verbalizer.type_name)

        if "query_type" in kwargs:
            query_type: type[Query] = kwargs["query_type"]
            if not issubclass(query_type, Query):
                error_msg = f"Query type must be a subclass of Query. Got: {query_type}"
                raise ValueError(error_msg)
            self._query = query_type
        else:
            self._query = Query.get(config.axs.query.type_name)

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
            self.simulator.env.unwrapped,
        )
        if not isinstance(macro_actions, dict) and not all(
            isinstance(k, int) for k in macro_actions
        ):
            error_msg = (
                f"Macro actions must be a dictionary with "
                f"int agent ids as keys. Got: {macro_actions}"
            )
            raise ValueError(error_msg)

        context_dict = self._verbalizer.convert(
            observations,
            macro_actions,
            infos=infos,
            query=self._query,
            env=self._simulator.env.unwrapped,
            **self.config.axs.verbalizer.params,
        )

        query_prompt = self._query_prompt.fill(
            context_dict,
            user_prompt=user_prompt,
            macro_names=self._macro_action.macro_names,
        )
        messages.append({"role": "user", "content": query_prompt})

        n = 0
        n_max = self.config.axs.n_max
        explanation, prev_explanation = None, None
        while (
            n < n_max
            and self._distance(explanation, prev_explanation) > self.config.axs.delta
        ):
            logger.info("Explanation iteration %d: %s", n, explanation)

            query_output = self._llm.chat(messages)[0]
            messages.append(query_output)
            query_content = query_output["content"]
            try:
                simulation_query = self._query.parse(query_content)
            except ValueError as e:
                logger.info(
                    "LLM-generated query was not valid. Query: %s",
                    query_content,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"The generated query is invalid. See error message: {e}"
                        ),
                    },
                )
                continue

            try:
                simulation_results = self._simulator.run(
                    simulation_query,
                    observations,
                    actions,
                    infos,
                )
                self._episodic_memory.learn(simulation_results)
            except SimulationError as e:
                error_msg = f"The simulation failed with error: {e}"
                logger.info(error_msg)
                messages.append({"role": "user", "content": error_msg})

            # Explanation synthesis
            simulation_context = self._verbalizer.convert(
                self._simulator.env.unwrapped,
                **simulation_results,
                **self.config.axs.verbalizer.params,
            )

            explanation_prompt = self._explanation_prompt.fill(simulation_context)
            messages.append(explanation_prompt)

            explanation_output = self._llm.chat(messages)[0]
            explanation = explanation_output["content"]
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
            # "simulator": self._simulator,
        }

        with Path(path).open("wb") as f:
            pickle.dump(statedict, f)

    def load_state(self, path: str) -> None:
        """Load the agent's state from a file except for the LLM."""
        with Path(path).open("rb") as f:
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
