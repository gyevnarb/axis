"""Run various baselines for AXS agent evaluation."""

import datetime
import logging
import pickle
from pathlib import Path

from util import get_agent, get_params

import axs
from envs import axs_igp2

SCENARIO = 1
USE_INTERROGATION = True
USE_CONTEXT = True

logger = logging.getLogger(__name__)
axs.util.init_logging(
        level="DEBUG",
        warning_only=[
            "igp2.core.velocitysmoother",
            "matplotlib",
            "httpcore",
            "openai",
            "httpx",
        ],
        log_dir=F"output/igp2/scenario{SCENARIO}/logs",
        log_name="test_prompt_only",
    )


params = get_params(
    scenarios=[SCENARIO],
    complexity=[1, 2],
    models=["llama-70b"],
    use_interrogation=USE_INTERROGATION,
    use_context=USE_CONTEXT,
    n_max=10 if USE_INTERROGATION else 0,
)

scenario_config = axs.Config(F"data/igp2/configs/scenario{SCENARIO}.json")
env = axs.util.load_env(scenario_config.env, scenario_config.env.render_mode)
env.reset(seed=scenario_config.env.seed)
agent_policies = axs.registry.get(scenario_config.env.policy_type).create(env)
logger.info("Created environment %s", scenario_config.env.name)

save_file = Path(scenario_config.output_dir, "agents", "agent_ep0.pkl")
start_dt = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")

results = []
for param in params:
    config = param.pop("config")

    axs_agent = axs.AXSAgent(config, agent_policies)
    prompt = axs.Prompt(**config.axs.user_prompts[1])

    truncations = [True]
    if not USE_INTERROGATION:
        truncations.append(False)

    for truncate in truncations:
        param["truncate"] = truncate
        logger.info(param)

        # Load the state of the agent from the file
        axs_agent.load_state(save_file)

        # Truncate the semantic memory until the current time
        if truncate and prompt.time is not None:
            semantic_memory = axs_agent.semantic_memory.memory
            for key in axs_agent.semantic_memory.memory:
                semantic_memory[key] = semantic_memory[key][: prompt.time]

        # Generate explanation to prompt
        user_query = prompt.fill()
        explanation, _ = axs_agent.explain(user_query)

        # Save results
        results.append(
            {
                "param": param,
                "config": config,
                "truncate": truncate,
                "explanation": explanation,
            },
        )

        save_name = f"test_prompt_only_{start_dt}.pkl"
        save_path = Path(config.output_dir, "results", save_name)
        with save_path.open("wb") as f:
            pickle.dump(results, f)
            logger.info("Results saved to %s", save_path)
