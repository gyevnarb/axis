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
