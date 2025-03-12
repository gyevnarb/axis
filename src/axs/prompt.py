""" This module contains the prompt generator functions for the LLM. """


def query_prompt(context) -> str:
    """ Prompt the LLM for a query to the simulator."""
    return "What is the query?"


def explanation_prompt(context) -> str:
    """ Prompt the LLM for an explanation."""
    return "What is the explanation?"
