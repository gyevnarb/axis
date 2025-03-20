"""Query implementation for IGP2."""

from typing import ClassVar

import axs

from .macroaction import IGP2MacroAction


class IGP2Query(axs.Query):
    """Query implementation for IGP2."""

    args_and_types: ClassVar[dict[str, dict[str, type]]] = {
        "add": {"location": tuple[float, float], "goal": tuple[float, float]},
        "remove": {"vehicle": int},
        "whatif": {"vehicle": int, "actions": list[IGP2MacroAction], "time": int},
        "what": {"vehicle": int, "time": int},
    }

    @classmethod
    def query_descriptions(cls) -> dict[str, str]:
        """Return a string with the query descriptions.

        You may refer query variables in your description using the format <variable>.
        These descriptions are used to generate the user prompt for the LLM.
        """
        return {
            "add": "What would happen if a new vehicle was present at <location> with goal <goal> from the start?",  # noqa: E501
            "remove": "What would happen if <vehicle> was removed from the road?",
            "whatif": "What would happen if <vehicle> took <actions> starting from <time>?",  # noqa: E501
            "what": "What will <vehicle> be doing at <time>?",
        }

    @classmethod
    def query_type_descriptions(cls) -> dict[str, str]:
        """Return a string with the query type descriptions.

        If not overriden, then descriptions are generated from the args_and_types
        automatically. The descriptions are used to generate user prompts for the LLM.
        """
        return {
            "location": "2D coordinate",
            "goal": "2D coordinate",
            "vehicle": "int",
            "actions": "list of macro actions from {macro_names}.",
            "time": "integer",
        }
