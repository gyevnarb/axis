"""Contains the main agent class for the AXS framework."""

import datetime
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
from axs.query import Query, QueryError
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
        self._prompts = {k: Prompt(v) for k, v in config.axs.prompts.items()}

        # Memory components
        cache_file = None
        if config.axs.cache_dir is not None:
            cache_dir = Path(config.axs.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            date_time = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
            cache_file = cache_dir.joinpath(f"memory_{date_time}.pkl")
        self._semantic_memory = SemanticMemory(
            {"observations": [], "actions": [], "infos": [], "prompts": []},
            cache=cache_file,
        )
        self._episodic_memory = EpisodicMemory(cache=cache_file)

        # Procedural components
        self._simulator = Simulator(config.env, agent_policies, simulator_env)
        self._llm = LLMWrapper(config.llm)
        self._distance = None

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

        self.semantic_memory.learn(prompts=user_prompt)
        self.episodic_memory.reset()

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

        context_dict = self._verbalizer.convert(
            observations,
            macro_actions,
            infos,
            None,
            self._query,
            self._simulator.env.unwrapped,
            **self.config.axs.verbalizer.params,
        )

        system_prompt = self._prompts["system"].fill(
            n_max=self.config.axs.n_max,
        )
        logger.debug("System prompt: %s", system_prompt)
        self.episodic_memory.learn(LLMWrapper.wrap("system", system_prompt))

        context_prompt = self._prompts["context"].fill(
            context_dict,
            macro_names=self._macro_action.macro_names,
            user_prompt=user_prompt,
        )
        logger.debug("Context prompt: %s", context_prompt)
        self.episodic_memory.learn(LLMWrapper.wrap("user", context_prompt))

        n = 1
        n_max = self.config.axs.n_max
        explanation, prev_explanation = None, None
        while (
            n <= n_max
            # and self._distance(explanation, prev_explanation) > self.config.axs.delta
        ):
            logger.info("Explanation iteration %d/%d", n, n_max)

            simulation_results = self._interrogate(
                observations,
                actions,
                macro_actions,
                infos,
                n,
            )
            explanation = self._explanation(user_prompt, simulation_results, n)
            if simulation_results == "DONE":
                break

            prev_explanation = explanation
            n += 1

        return explanation

    def _interrogate(
        self,
        observations: Any,
        actions: list[Any],
        macro_actions: dict[int, MacroAction],
        infos: list[dict[str, Any]],
        n: int = 0,
    ) -> dict[str, Any] | str:
        """Perform a single round of simulator interrogation with the LLM.

        Returns:
            dict[str, Any] | str: The simulation results or a string indicating
                if the simulation is done or failed.

        """
        interrogation_prompt = self._prompts["interrogation"].fill(
            n=n,
            n_max=self.config.axs.n_max,
        )
        self.episodic_memory.learn(LLMWrapper.wrap("user", interrogation_prompt))

        for _ in range(self.config.axs.n_tries):
            query_output = self._llm.chat(self.episodic_memory.memory)[0]
            self.episodic_memory.learn(query_output)
            query_content = query_output["content"]
            if "DONE" in query_content:
                return "DONE"

            try:
                simulation_query = self._query.parse(query_content)
                simulation_query.verify(
                    self._simulator.env.unwrapped,
                    observations[-1],
                    actions[-1],
                    macro_actions,
                    infos[-1],
                )

                simulation_results = self._simulator.run(
                    simulation_query,
                    observations,
                    infos,
                )
                break

            except QueryError as e:
                error_msg = (
                    f"The generated query is invalid: {e} Generate a different query."
                )
            except SimulationError as e:
                error_msg = f"The simulation failed: {e} Generate a different query."
            logger.exception(error_msg)
            self.episodic_memory.learn(LLMWrapper.wrap("user", error_msg))

        else:
            error_msg = "The simulation failed after multiple attempts."
            raise RuntimeError(error_msg)

        return simulation_results

    def _explanation(
        self,
        user_prompt: str,
        results: dict[str, Any],
        n: int,
    ) -> str:
        """Synthesise an explanation based on the simulation results."""
        if results == "DONE":
            explanation_prompt = self._prompts["final"].fill()
        else:
            simulation_context = self._verbalizer.convert(
                **results,
                query=None,
                env=self._simulator.env.unwrapped,
                **self.config.axs.verbalizer.params,
            )

            explanation_prompt = self._prompts["explanation"].fill(
                simulation_context,
                user_prompt=user_prompt,
                n=n,
                n_max=self.config.axs.n_max,
            )
        self.episodic_memory.learn(LLMWrapper.wrap("user", explanation_prompt))

        explanation_output = self._llm.chat(self.episodic_memory.memory)[0]
        self.episodic_memory.learn(explanation_output)

        return explanation_output["content"]

    def reset(self) -> None:
        """Reset the agent."""
        self._semantic_memory.reset()
        self._episodic_memory.reset()

    def save_state(self, path: str) -> None:
        """Save the agent's state to a file except the LLM."""
        statedict = {
            "semantic_memory": self._semantic_memory,
            "episodic_memory": self._episodic_memory,
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
