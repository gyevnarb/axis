"""Run various baselines for AXS agent evaluation."""

import datetime
import logging
import pickle
from pathlib import Path

from util import get_agent, get_configs

import axs
from envs import axs_igp2

logger = logging.getLogger(__name__)
output_dir = Path("output", "igp2")

configs = get_configs(
    scenarios=[1],
    complexity=[1, 2],
    models=["llama-70b"],
    use_interrogation=False,
    use_context=True,
    n_max=10,
)

results = []
for config in configs:
    logger.info(config.axs.prompts)

    save_file = Path(config.output_dir, "agents", "agent_ep0.pkl")

    axs_agent = get_agent(config)

    # Iterate over all save files
    start_dt = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")

    prompt = axs.Prompt(**config.axs.user_prompts[1])

    # Load the state of the agent from the file
    axs_agent.load_state(save_file)
    logger.info("Loaded state from %s", save_file)

    # Truncate the semantic memory until the current time
    for truncate in [True, False]:
        if truncate and prompt.time is not None:
            semantic_memory = axs_agent.semantic_memory.memory
            for key in axs_agent.semantic_memory.memory:
                semantic_memory[key] = semantic_memory[key][: prompt.time]

        user_query = prompt.fill()
        explanation, _ = axs_agent.explain(user_query)

        results.append(
            {
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
