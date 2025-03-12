from typing import List


class MacroAction:

    @classmethod
    def wrap(cls, actions) -> List["MacroAction"]:
        pass

    def unwrap(self, macro_actions) -> List:
        pass
