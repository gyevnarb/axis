"""Playground script for testing."""

import axs

from envs import axs_igp2

config = axs.Config("data/igp2/configs/scenario1.json")
agent = axs.AXSAgent(config)
agent.load_state("output/igp2/scenario1/results_20250404_140413.pkl")

sem = agent.semantic_memory
sem.memory[-2] = axs.
