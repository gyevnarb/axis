"""Playground script for testing."""

import axs
from envs import axs_igp2

config = axs.Config("data/igp2/configs/scenario2.json")
agent = axs.AXSAgent(config)
agent.load_state("output/igp2/scenario2/agent_ep0.pkl")

final_prompt = """Generate your final explanation using all previous information in response to the question: {user_prompt}
Do not include state or action descriptions. Be specific and concise, focusing on causal relationships.""".format(user_prompt="What if vehicle 1 hadn't changed lanes right?")

sem = agent.semantic_memory
sem.memory["messages"][-2] = axs.LLMWrapper.wrap("user", final_prompt)
sem.memory["messages"].pop(-1)
response = agent.llm.chat(sem.memory["messages"])
print("Response:", response[0][0]["content"])
 