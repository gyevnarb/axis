""" The axs package is package that allows the user to generate explanation for any gymnasium environment with an agentic workflow. """
import typer
from typing import Annotated

from vllm import LLM
from vllm import SamplingParams

from axs import prompt
from axs.verbalize import Verbalizer
from axs.macro_action import MacroAction
from axs.memory import SemanticMemory, EpisodicMemory
from axs.simulator import Simulator


app = typer.Typer()


@app.callback()
def main(
    user_query : Annotated[
        str,
        typer.Argument(help="The query to be explained.")
    ],
    model : Annotated[
        str,
        typer.Option(help="LLM model name as specified on HuggingFace.")
    ] = "llama-1B",
    n_max: Annotated[
        int,
        typer.Option(help="The maximum number of iterations for explanation generation.")
    ] = 5,
    delta: Annotated[
        float,
        typer.Option(help="Convergence threshold for minimum distance between consecutive explanations.")
    ] = 0.01,
    sampling_params: Annotated[
        dict,
        typer.Option(help="Sampling parameters for the LLM.")
    ] = {}
) -> None:
    """ Call the typer app. """
    llm = LLM(model)
    simulator = Simulator("gofi-v1")
    verbalizer = Verbalizer()
    semantic_memory = SemanticMemory()
    episodic_memory = EpisodicMemory()

    messages = [
        {"role": "system", "content": "TODO: Add system message."}
    ]

    states = semantic_memory.retrieve("states")
    actions = semantic_memory.retrieve("actions")
    macro_actions = MacroAction.wrap(actions)

    txt_states = verbalizer.convert(states)
    txt_actions = verbalizer.convert(macro_actions)

    query_prompt = prompt.query_prompt(user_query, txt_states, txt_actions)
    messages.append(query_prompt)

    n = 0
    explanation, prev_explanation = None, None
    sampling_params = SamplingParams(**sampling_params)
    while n < n_max and distance(explanation, prev_explanation) > delta:
        output = llm.chat(messages)
        simulation_query = 
        prev_explanation = explanation
        n += 1


    app()


if __name__ == "__main__":
    app()