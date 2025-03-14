from typing import List


class MacroAction:

    @classmethod
    def wrap(cls, actions) -> List["MacroAction"]:
        pass

    def unwrap(self, macro_actions) -> List:
        pass


class IGP2MacroAction(MacroAction):
    pass


class MacroActionFactory:

    macro_actions = {
        "igp2": IGP2MacroAction
    }

    @classmethod
    def create(cls, config) -> MacroAction:
        """ Create a macro action based on its name and config. """
        return cls.macro_actions[config.name](config)