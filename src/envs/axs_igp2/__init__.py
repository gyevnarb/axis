""" This module contains implementation of AXS for IGP2. """
from axs import Verbalizer, MacroAction

from .macroaction import IGP2MacroAction
from .verbalizer import IGP2Verbalizer

__all__ = [
    "IGP2MacroAction",
    "IGP2Verbalizer"
]

MacroAction.register('IGP2MacroAction', IGP2MacroAction)
Verbalizer.register('IGP2Verbalizer', IGP2Verbalizer)
