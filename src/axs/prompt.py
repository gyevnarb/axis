"""Module containing prompt generator functions for LLMs."""

from string import Formatter
from typing import Any


class SafeDict(dict):
    """A dictionary that returns the key as a value if the key is missing."""

    def __missing__(self, key: str) -> str:
        """Return the key as a value if the key is missing."""
        return "{" + key + "}"


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

    def __repr__(self) -> str:
        """Return a string representation of the Prompt object."""
        placeholders = ", ".join(self.placeholders)
        return f"Prompt({placeholders})"

    def __str__(self) -> str:
        """Return the string representation of the Prompt object."""
        return repr(self)

    def __eq__(self, other: object) -> bool:
        """Check if two Prompt objects are equal."""
        if not isinstance(other, Prompt):
            return False
        return self.template == other.template and self.time == other.time

    def fill(
        self, context_dict: dict[str, str] | None = None, **content: dict[str, Any],
    ) -> str:
        """Complete the prompt from the template and context.

        Additional content can be passed as keyword arguments,
        which will be added to query text after the context has been initialized.

        Args:
            context_dict (dict): The context dictionary to fill the template.
            content (dict[str, Any]): Additional content to fill the template.`

        """
        if context_dict is None:
            context_dict = {}

        all_vars = {k for d in [context_dict, content] for k in d}
        for k in self.placeholders:
            if k not in all_vars:
                error_msg = f"Placeholder '{k}' not specified in context."
                raise ValueError(error_msg)

        temp_str = self.template.format_map(SafeDict(context_dict))
        return temp_str.format_map(SafeDict(content))

    @property
    def placeholders(self) -> list[str]:
        """Return a list of placeholders in the template."""
        fmt = Formatter()
        return [p[1] for p in fmt.parse(self.template) if p[1] is not None]
