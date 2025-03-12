from axs.memory import SemanticMemory, EpisodicMemory
from axs.verbalize import Verbalizer


class AXSAgent:
    def __init__(self):
        # Memory components
        self._semantic_memory = SemanticMemory()
        self._episodic_memory = EpisodicMemory()

        # Procedural components
        self._llm = None  # TODO: Add LLM model
        self._simulator = None  # TODO: Add simulator

        # Utility components
        self._verbalizer = Verbalizer()

    @property
    def semantic_memory(self):
        return self._semantic_memory

    @property
    def episodic_memory(self):
        return self._episodic_memory

    @property
    def llm(self):
        return self._llm

    @property
    def simulator(self):
        return self._simulator

    @property
    def verbalizer(self):
        return self._verbalizer