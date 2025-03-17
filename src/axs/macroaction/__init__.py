""" The module axs.macroaction provides base classes
and implementations for macro-actions.

To create your own macro action, inherit from the base class
MacroAction and implement the wrap and unwrap methods.
Subsequently, register your macro action with the
MacroActionFactory to be able to use it in the configuration file.
"""
from axs.macroaction.base import MacroAction, MacroActionFactory
from axs.macroaction.igp2 import IGP2MacroAction

MacroActionFactory.register('IGP2MacroAction', IGP2MacroAction)
