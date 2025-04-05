"""Implementation of AXS for IGP2."""

import logging
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

import axs
from axs import cli

from .macroaction import IGP2MacroAction
from .policy import IGP2Policy
from .query import IGP2Query
from .verbalize import IGP2Verbalizer
from .wrapper import IGP2QueryableWrapper

__all__ = [
    "IGP2MacroAction",
    "IGP2Policy",
    "IGP2Query",
    "IGP2QueryableWrapper",
    "IGP2Verbalizer",
]


logger = logging.getLogger(__name__)


class FunctionNames(str, Enum):
    """Function names for IGP2."""

    run = "run"
    evaluate = "evaluate"


@axs.app.command()
def igp2(
    ctx: typer.Context,
    function: Annotated[
        FunctionNames, typer.Argument(help="Function to run."),
    ],
    save_logs: Annotated[bool, typer.Option(help="Save logs to file.")] = False,
) -> None:
    """Run an AXS agent with the IGP2 configurations."""
    config = ctx.obj["config"]
    debug = config.debug
    output_dir = config.output_dir
    axs.util.init_logging(
        level="DEBUG" if debug else "INFO",
        warning_only=[
            "igp2" if not debug else "igp2.core.velocitysmoother",
            "matplotlib",
            "httpcore",
            "openai",
            "httpx",
        ],
        log_dir=Path(output_dir, "logs") if save_logs else None,
        log_name=function[:4],
    )

    # Call the function dynamically
    try:
        getattr(axs, function)(ctx)
    except Exception as e:
        logger.exception(
            "Error occurred while running the function %s", function, exc_info=e,
        )
