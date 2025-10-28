# Codebase for Agentic Explanations via Interrogative Simulations (AXIS)

This file is the documentation for the AXIS codebase. The folder hierarchy of the codebase is given at the bottom of this README under section [Hierarchy](#hierarchy).

_Note 1: For legacy reasons, the package command names in the codebase are `axs` rather than ~~`axis`~~._

_Note 2: This is a research project. The code here has not been extensively tested for all edge-cases and bugs, and it is likely not the most efficient implementation of the underlying ideas. If you find any problems that are hard to debug, please open a new issue on Github._

## Table of Contents
- [Installation](#installation)
- [AXIS Structure and Implementation](#axis-structure-and-implementation)
- [Reproducing IGP2 results](#reproducing-axis-experiments-with-igp2)
- [Folder hierarchy](#hierarchy)
- [Citation](#citation)

## Installation

To use our codebase, you must install the necessary dependencies. The below instructions assume the use of `uv` for package installation and command running.

#### 0. Install `uv`:
The software `uv` is a manager for Python environments and projects and is orders of magnitued faster than `pip`.
To install `uv`, please follow the instructions on their webpage [here](https://docs.astral.sh/uv/getting-started/installation/).

#### 1. Install AXIS:
From the root folder, run the command `uv sync` to download and create a virtual environment for running the approporiate commands. If you wish to run LLMs localy using vLLM then instead run the command `uv sync --extra vllm`. If you wish to reproduce the results of the paper using [IGP2](https://github.com/uoe-agents/IGP2) then use the command `uv sync --extra igp2`.

#### 2. Test it works:
To make sure that your install has succeeded, from the you can now run the command `uv run axs --help`, which should return the help message for the `axs` command.

## AXIS Structure and Implementation

This section describes how AXIS is structured and how to implement AXIS for your own use case.
We assume in our implementation that the simulator is defined using [`gymnasium`](https://github.com/Farama-Foundation/Gymnasium) or [`pettingzoo`](https://github.com/Farama-Foundation/PettingZoo).

_Note: we haven't extensively tested the pettingzoo implementation, so bugs are likely._

AXIS is defined in an object-oriented way using heavy polymorphism, with abstract classes defining high-level procedural patterns, and subclasses implementing the actual logic.
To get an idea of how to override the functions in the abstract classes, you can browse the `src/axs_igp2` folder, which contains the implementation for the IGP2-based experiments.

The backbone of AXIS is defined in the `src/axs` folder. The main files to pay attention to are:
- `agent.py`: The core implementation of the AXIS algorithm (Algorithm 1. in the paper). Ideally, you will not have to touch this file.
- `macroaction.py`: Abstract class for macro actions. Inherit the abstract class `MacroAction` to define your own macro actions. The field `MacroAction.macro_names` should define the names of your macro actions, while the rest of the methods should define the procedures to convert action sequences to macro actions and back.
- `policy.py`: Abstract class for wrapping trained agent policies to work with AXIS. Inherit the abstarct class `Policy` to enable interfacing between AXIS and the simulator.
- `prompt.py`: Contains the class `Prompt` which parses prompt templates and, when requested, formats them with the given data. You can define your own prompts using standard Python formatting syntax, placing the prompt templates into the folder `data/xxx/prompts`, where xxx is your project's name.
- `query.py`: Contains the class `Query` which defines the four interrogation queries used in AXIS. Inherit this class to define how queries are used in the simulator.
- `verbalize.py`: Defines the abstract class `Verbalizer` which describes the methods used for verbalizing environmental representations. Inherit the class `Verbalizer` to define your own verbalization logic.
- `wrapper.py`: The classes in this file act as the communication channel between AXIS and the external simulator. Inherit the class `QueryableWrapper` or `QueryableAECWrapper` to define how queries are applied to the simulation and how the simulation-returned data is processed for AXIS.


The following files are also a part of the implementation, but are less important to adapting AXIS to a new domain:
- `config.py`: The configuration class hiearchy, used to parse configuration files. You can either read the class hiearchy in this file to understand how to structure your JSON configs, or look at example configs in the `data/igp2/configs` folder. Ideally, you will not have to touch this file, but you can extend the abstract `ConfigBase` class to define your own configuration setup.
- `llm.py`: A lightweight LLM wrapper to support interaction with hosted (either locally or online) LLMs.
- `memory.py`: Memory modules to store different longer term data.
- `simulator.py`: Defines the `Simulator` class which provides a fixed API to interact with external simulators.


## Reproducing AXIS Experiments with IGP2
Please follow the below steps to reproduce our work.

#### 1. Run the scenario/scenarios

Before generating explanations for the scenarios, you must run the scenarios to save the scenario execution data to disk.
You can do this by calling the command `uv run axs-igp2 --help` and specifying the command line arguments as necessary.
Scenario configuration files are found under the `data/igp2/configs/` folder.
Results will be saved into an `output/` folder.

**Example:** To generate a scenario run for scenario #1, run the following command: `uv run axs-igp2 -c data/igp2/configs/scenario1.json --save-results run`.

#### 2. (Optional) Run all scenarios and evaluations automatically

If you prefer to just start a script and leave it running, you can run the below scripts with various levels of customization:

1. Basic usage (with defaults):
`bash scripts/bash/generate_all.sh` followed by `bash scripts/bash/evaluate_all.sh`
    - GPT-4.1 for generation
    - Claude 3.5 for evaluation
    - With interrogation and context
    - Evaluating only "final" explanations
    - Specifying a generation model:

2. To specify a generation / evaluation model, for exapmle use:
`bash scripts/bash/generate_all.sh llama70b` and `bash scripts/bash/evaluate_all.sh qwen72b`

3. To specify whether to use interrogation or to test all feature combinations use:
`bash scripts/bash/generate_all.sh llama70b --no-interrogation --use-context"` and similarly for the evaluation script.

The bash script experiments call Python scripts located in the `scripts/python` directory.
These Python scripts may also be called individually, following their Command Line Interface (CLI).
The below stages 1-4 explains how to do this, in order to run things on a controlled, step-by-step level:

#### 3A. Decide which LLM to use:
In order to run experiments, you need to specify the LLM to use. The file [`llm_configs.json`](scripts/python/llm_configs.json) contains all LLMs used in the paper, and you may add your own to this file. It currently contains the following LLMs:

- gpt41 (GPT-4.1)
- gpt4o (GPT-4o)
- o4mini (o4 Mini)
- gpt41mini (GPT-4.1 Mini)
- deepseekv3 (DeepSeek Chat)
- deepseekr1 (DeepSeek Reasoner)
- claude35 (Claude 3.5 Haiku)
- claude37 (Claude 3.7 Sonnet)
- llama70b (Llama 3.3-70B Instruct)
- qwen72b (Qwen 2.5-72B Instruct)

If using an API-based LLM, you must set the API key in the corresponding environment variable. We use the standard API variable names of each company, which you can also find in the [`llm_configs.json`](scripts/python/llm_configs.json).

#### 3B. Generate explanations

You can perform this step once the scenario data is saved for the scenario for which you would like to generate explanations.
To generate explanations for a given LLM, scenario, and user prompt, you can run the command `uv run python scripts/python/generate.py --help` and specify the command line arguments as necessary.

**Example:** To generate explanations for scenario #1 with GPT-4.1 for all prompts with the vebalised features macro actions and observations and a concise linguistic complexity, run: `uv run python scripts/python/generate.py -s 1 -m gpt41 -c 1 --features '["add_macro_actions", "add_observations"]'`.

Results will be saved into the `output/` folder, and will override existing results there.
The save name will contain information about the specific arguments used for generating explanations.
For example, the file name `gpt41_interrogation_context.pkl` means that the GPT-4.1 model was used with interrogagtion and initial context given.

*Note: This command assumes that either the API key is given for the selected LLM, or that the LLM is running on localhost.*

#### 4. Evaluate explanations

Once explanations have been generated for a given scenario, you can evaluate it with an LLM by calling `uv run python scripts/python/evaluate.py --help` and specifyin the command line arguments as necessary.

**Example:** To evaluate all results for the above generation command with Claude 3.5, run: `uv run python scripts/python/evaluate.py -s 1 -m claude35 -r gpt41_interrogation_context.pkl -e all`.

Results will be saved to the `output/` folder, and the file name will contain the word 'evaluate' and the name of the LLM used for evaluation.
For example, the name `evaluate_gpt41_claude35_interrogation_context.pkl` contains the evaluation results for the above explanation file and using Claude 3.5 as an external reward model.

#### 5. Analysis and Generate Figures

Finally, using the generated explanations and their evaluations, we can recreate the plots and tables from the paper by running the command `uv run python/scripts/analysis.py --help` and specifying the command line arguments as necessary.
The results will be saved into the `output/analysis/` or `output/plots/` folders.

To inspect the generated explanations, the command `uv run python/scripts/summary.py --help` may also be used.

## Hierarchy

Below is the folder hierarchy of the AXIS codebase.

```
src/
├── axs/                    # Abstract classes and algorithm implementations; may be overriden for specialising to environments
├── envs/                   # Environment implementations overriding abstract classes in the axs folder
│   └── axs_igp2/           # AXIS implementation for IGP2 and GOFI autonomous driving environments
│
├── scripts/
│   ├── bash/                   # Shell scripts for running experiments
│   │   ├── TODO.py            # Scoring and ranking scripts
│   │   └── TODO2.py          # Summary generation for generation and evaluation result files
│   │
│   └── python/                 # Python scripts for analysis and visualization with IGP2
│       ├── analysis.py         # Data analysis utilities
│       ├── evaluate.py         # Explanation evaluation scripts
│       ├── generate.py         # Explanation generation script
│       ├── plot.py             # Plotting utilities (shapley waterfall, barplots, etc.)
│       ├── score.py            # Scoring and ranking scripts
│       └── summary.py          # Summary generation for generation and evaluation result files
│
data/
├── igp2/                   # Data for IGP2 autonomous driving scenarios
│   ├── configs/            # Configuration files for scenarios
│   ├── evaluation/         # Evaluation data and prompts for external reward model
│   ├── maps/               # Map definitions using OpenDRIVE standard
│   └── prompts/            # AXIS prompt templates for IGP2
│
output/                     # (Only present if generation/evaluation has been done)
├── igp2/                   # Output directory for IGP2 results
│   ├── plots/              # Generated plots and visualizations
│   └── scenarioN/...       # Results for each scenario (N in 0-9)
│       ├── agents/         # Save files for agents and explanation cache
│       ├── logs/           # Logging files for the scenario
│       └── results/        # Results files for each scenario
```

## Citation

If you use or build on our codebase, please cite the following pre-print:

```latex
@misc{gyevnar2025axis,
  title = {Integrating Counterfactual Simulations with Language Models for Explaining Multi-Agent Behaviour},
  author = {Gyevnár, Bálint and Lucas, Christopher G. and Albrecht, Stefano V. and Cohen, Shay B.},
  year = {2025},
  month = may,
  url = {https://arxiv.org/abs/2505.17801},
}
```