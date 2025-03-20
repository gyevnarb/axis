"""Module containing prompt generator functions for LLMs."""

from string import Formatter
from typing import Any


class Prompt:
    """A class to generate prompts from templates and context.

    The template should be specified with placeholders for the context variables
    such that they can be formatted using the `str.format` method.
    """

    def __init__(self, template: str, time: int | None = None) -> "Prompt":
        """Initialize Prompt with template and optional timestep for when it is valid.

        Args:
            template (str): The template string with placeholders for context variables.
            time (int): The time step when the prompt becomes valid

        """
        self.template = template
        self.time = time

    def fill(self, **content: dict[str, Any]) -> str:
        """Complete the prompt from the template and context."""
        for k in content:
            if k not in self.placeholders:
                error_msg = f"Placeholder '{k}' not specified in context."
                raise ValueError(error_msg)
        return self.template.format(**content)

    @property
    def placeholders(self) -> list[str]:
        """Return a list of placeholders in the template."""
        fmt = Formatter()
        return [p[1] for p in fmt.parse(self.template) if p[1] is not None]
