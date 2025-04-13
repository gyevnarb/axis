"""Run various baselines for AXS agent evaluation."""

import datetime
import logging
import pickle
from pathlib import Path

import axs
from envs import axs_igp2

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("output", "igp2")

save_folders = list(Path(OUTPUT_DIR).glob("scenario*"))
if not save_folders:
    error_msg = f"No save files found in {OUTPUT_DIR}"
    raise FileNotFoundError(error_msg)

for save_folder in save_folders:
    scenario_name = save_folder.name + ".json"
    config_file = Path("data", "igp2", "configs", scenario_name)
    config = axs._init_axs(config_file, None, None, None, None)
    config.config_dict["axs"]["use_interrogation"] = False
    print(config.axs.prompts)

    save_files = list(Path(save_folder, "agents").glob("agent_ep*.pkl"))
    if not save_files:
        error_msg = f"No save files found in {save_folder}"
        raise FileNotFoundError(error_msg)

    env = axs.util.load_env(config.env, config.env.render_mode)
    env.reset(seed=config.env.seed)
    logger.info("Created environment %s", config.env.name)

    agent_policies = axs.registry.get(config.env.policy_type).create(env)
    axs_agent = axs.AXSAgent(config, agent_policies)

    # Iterate over all save files
    start_dt = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
    results = {}
    for ep_ix, save_file in enumerate(save_files):
        ep_results = {}

        # Run all explanations.
        for p_ix, prompt_dict in enumerate(config.axs.user_prompts):
            prompt = axs.Prompt(**prompt_dict)

            # Load the state of the agent from the file
            axs_agent.load_state(save_file)
            logger.info("Loaded state from %s", save_file)

            # Truncate the semantic memory until the current time
            if prompt.time is not None:
                semantic_memory = axs_agent.semantic_memory.memory
                for key in axs_agent.semantic_memory.memory:
                    semantic_memory[key] = semantic_memory[key][:prompt.time]

            user_query = prompt.fill()
            context, _ = axs_agent.explain(user_query, context_only=True)

            messages = [
                axs.LLMWrapper.wrap("system", )
            ]

            if config.save_results:
                save_name = f"checkpoint_{start_dt}_p{p_ix}.pkl"
                save_path = Path(config.output_dir, "checkpoints", save_name)
                with save_path.open("wb") as f:
                    pickle.dump(p_results, f)
                    logger.info("Episode %d checkpoint saved to %s", ep_ix, save_path)
            ep_results[f"p{p_ix}"] = p_results