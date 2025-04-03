"""Contains the main agent class for the AXS framework."""

import datetime
import logging
import pickle
from copy import copy, deepcopy
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
        if not config.axs.use_context:
            self._prompts["context"] = self._prompts["no_context"]

        # Memory components
        save_dir = None
        if config.save_results:
            save_dir = Path(config.output_dir, "cache")
            save_dir.mkdir(parents=True, exist_ok=True)
            date_time = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
        self._semantic_memory = SemanticMemory(
            {"observations": [], "actions": [], "infos": []},
            save_file=save_dir.joinpath(f"semantic_{date_time}.pkl")
            if save_dir
            else None,
        )
        self._episodic_memory = EpisodicMemory(
            save_file=save_dir.joinpath(f"episodic_{date_time}.pkl")
            if save_dir
            else None,
        )
        self._query_memory = EpisodicMemory()

        # Procedural components
        self._agent_policies = agent_policies
        self._simulator = Simulator(config.env, agent_policies, simulator_env)
        self._llm = LLMWrapper(config.llm)
        self._distance = None

        # Utility components
        if "macro_type" in kwargs:
            macro_type: type[MacroAction] = kwargs["macro_type"]
            if not issubclass(macro_type, MacroAction):
                error_msg = (
                    f"Macro type must be a subclass of MacroAction. Got: {macro_type}."
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
                    f"Got: {verbalizer_type}."
                )
                raise ValueError(error_msg)
            self._verbalizer = verbalizer_type
        else:
            self._verbalizer = Verbalizer.get(config.axs.verbalizer.type_name)

        if "query_type" in kwargs:
            query_type: type[Query] = kwargs["query_type"]
            if not issubclass(query_type, Query):
                error_msg = f"Query type must be subclass of Query. Got: {query_type}."
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

        # Turn on saving the episodic and semantic memory on each learn call.
        self.semantic_memory.saving = True
        self.episodic_memory.saving = True

        # Reset the internal episodic memory from previous calls and save user prompt.
        self.semantic_memory.learn(prompts=user_prompt)
        self.episodic_memory.reset()

        observations = self._semantic_memory.retrieve("observations")
        actions = self._semantic_memory.retrieve("actions")
        infos = self._semantic_memory.retrieve("infos")

        # Convert actions and observations to higher-level macro actions.
        macro_actions = self._macro_action.wrap(
            self.config.axs.macro_action,
            actions,
            observations,
            infos,
            self.simulator.env.unwrapped,
        )

        # Convert observations and macro actions to a context dictionary of strings
        # that will be passed to the LLM directly.
        context_dict = self._verbalizer.convert(
            observations,
            macro_actions,
            infos,
            None,
            self._query,
            self._simulator.env.unwrapped,
            **self.config.axs.verbalizer.params,
        )

        # Create system prompt
        system_prompt = self._prompts["system"].fill(
            n_max=self.config.axs.n_max,
        )
        logger.debug("System prompt: %s", system_prompt)
        self.episodic_memory.learn(LLMWrapper.wrap("system", system_prompt))

        # Create context prompt
        context_prompt = self._prompts["context"].fill(
            context_dict,
            macro_names=self._macro_action.macro_names,
            user_prompt=user_prompt,
        )
        logger.debug("Context prompt: %s", context_prompt)
        self.episodic_memory.learn(LLMWrapper.wrap("user", context_prompt))

        n = 1
        n_max = self.config.axs.n_max
        distance = float("inf")
        explanation, prev_explanation = None, None
        statistics = {
            "n_tries": [],
            "n": n,
            "distances": [],
            "simulation_results": [],
            "usage": [],
        }
        while n <= n_max and distance > self.config.axs.delta:
            logger.info("Explanation iteration %d/%d", n, n_max)

            # Interrogation stage
            simulation_results = self._interrogate(
                observations,
                actions,
                macro_actions,
                infos,
                statistics,
            )
            statistics["simulation_results"].append(simulation_results)

            # Explanation stage
            explanation = self._explanation(user_prompt, simulation_results, statistics)
            if simulation_results == "DONE":
                break

            prev_explanation = explanation
            distance = float("inf")  # self._distance(explanation, prev_explanation)
            n += 1

            # Store statistics of the iteration
            statistics["n"] = n
            statistics["distances"].append(distance)

        self.semantic_memory.learn(
            messages=copy(self.episodic_memory.memory),
            explanations=explanation,
            statistics=statistics,
        )
        logger.info("Final explanation: %s", explanation)

        # Turn off constantly saving the episodic and semantic memory.
        self.semantic_memory.saving = False
        self.episodic_memory.saving = False

        return explanation

    def _interrogate(
        self,
        observations: Any,
        actions: list[Any],
        macro_actions: dict[int, MacroAction],
        infos: list[dict[str, Any]],
        statistics: dict[str, Any],
    ) -> dict[str, Any] | str:
        """Perform a single round of simulator interrogation with the LLM.

        Returns:
            dict[str, Any] | str: The simulation results or a string indicating
                if the simulation is done or failed.

        """
        interrogation_prompt = self._prompts["interrogation"].fill(
            n=statistics["n"],
            n_max=self.config.axs.n_max,
        )
        self.episodic_memory.learn(LLMWrapper.wrap("user", interrogation_prompt))

        for n_tries in range(1, self.config.axs.n_tries + 1):  # noqa: B007
            query_outputs, usage = self._llm.chat(self.episodic_memory.memory)
            query_output = query_outputs[0]
            query_content = query_output["content"]

            self.episodic_memory.learn(query_output)
            statistics["usage"].append(usage)

            if query_content == "DONE":
                return "DONE"

            try:
                simulation_query = self._query.parse(query_content)
                simulation_query.verify(
                    self._simulator.env.unwrapped,
                    observations,
                    actions,
                    macro_actions,
                    infos,
                )

                # Check whether the query has been called before
                if simulation_query in self._query_memory.memory:
                    error_msg = (f"The query {query_content} was already tested. "
                                f"Generate a different query.")
                    self.episodic_memory.learn(LLMWrapper.wrap("user", error_msg))
                    continue
                self._query_memory.learn(deepcopy(simulation_query))

                simulation_results = self._simulator.run(
                    simulation_query,
                    observations,
                    actions,
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

        statistics["n_tries"].append(n_tries)
        return simulation_results

    def _explanation(
        self,
        user_prompt: str,
        results: dict[str, Any],
        statistics: dict[str, Any],
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
                n=statistics["n"],
                n_max=self.config.axs.n_max,
            )
        self.episodic_memory.learn(LLMWrapper.wrap("user", explanation_prompt))
        logger.debug("Explanation prompt: %s", explanation_prompt)

        explanation_outputs, usage = self._llm.chat(self.episodic_memory.memory)
        explanation_output = explanation_outputs[0]
        self.episodic_memory.learn(explanation_output)
        statistics["usage"].append(usage)

        return explanation_output["content"]

    def reset(self) -> None:
        """Reset the agent."""
        self._semantic_memory.reset()
        self._episodic_memory.reset()

    def save_state(self, path: str | Path) -> None:
        """Save the agent's state to a file except the LLM."""
        statedict = {
            "semantic_memory": self._semantic_memory.memory,
            "episodic_memory": self._episodic_memory.memory,
        }
        with Path(path).open("wb") as f:
            pickle.dump(statedict, f)

    def load_state(self, path: str | Path) -> None:
        """Load the agent's state from a file except for the LLM."""
        with Path(path).open("rb") as f:
            statedict = pickle.load(f)
        self._semantic_memory._mem = statedict["semantic_memory"]
        self._episodic_memory._mem = statedict["episodic_memory"]

    @property
    def agent_policies(self) -> dict[int, Policy]:
        """The agents' policies."""
        return self._agent_policies

    @property
    def episodic_memory(self) -> EpisodicMemory:
        """The agent's episodic memory."""
        return self._episodic_memory

    @property
    def llm(self) -> LLMWrapper:
        """The agent's LLM model."""
        return self._llm

    @property
    def semantic_memory(self) -> SemanticMemory:
        """The agent's semantic memory."""
        return self._semantic_memory

    @property
    def simulator(self) -> Simulator:
        """The agent's simulator."""
        return self._simulator

    @property
    def verbalizer(self) -> Verbalizer:
        """The agent's verbalizer."""
        return self._verbalizer
