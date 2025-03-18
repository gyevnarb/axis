""" This module contains the prompt generator functions for the LLM. """
from string import Formatter
from typing import List


class Prompt:
    """ A class to generate prompts from templates and context.
    The template should be specified with placeholders for the context variables
    such that they can be formatted using the `str.format` method."""

    def __init__(self, template: str, time: int = None):
        """ Initialize the Prompt with the template with an
        optional time step for when it becomes valid.

        Args:
            template (str): The template string with placeholders for context variables.
            time (int): The time step when the prompt becomes valid
        """
        self.template = template
        self.time = time

    def fill(self, **context) -> str:
        """ Complete the prompt from the template and context. """
        for k in context:
            if k not in self.placeholders:
                raise ValueError(f"Placeholder {k} not specified in context.")
        return self.template.format(**context)

    @property
    def placeholders(self) -> List[str]:
        """ Return a list of placeholders in the template. """
        fmt = Formatter()
        return [p[1] for p in fmt.parse(self.template) if p[1] is not None]
