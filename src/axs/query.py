"""Define queries used in the Simulator with arguments and types."""

import re
from typing import Any, ClassVar

from axs.config import Registerable

QueryTypeMap = dict[str, dict[str, type]]
QueryTypeMap.__doc__ = "Dictionary mapping valid query names to arguments and types."


class Query(Registerable, class_type=None):
    """Represents queries with arguments used in Simulator.

    The name of the valid queries are fixed, but the arguments and types may
    be overriden to customize the query behavior.
    """

    valid_queries: ClassVar[list[str]] = [
        "add",
        "remove",
        "whatif",
        "what",
    ]
    args_and_types: ClassVar[QueryTypeMap]

    def __init__(self, name: str, params: dict[str, Any]) -> "Query":
        """Initialize new Query."""
        self.name = name
        self.params = params

    @classmethod
    def query_descriptions(cls) -> str:
        """Return a string with the query descriptions."""
        # TODO: Implement this

    @classmethod
    def query_type_descriptions(cls) -> str:
        """Return a string with the query type descriptions."""
        # TODO: Implement this

    @classmethod
    def parse(cls, query_str: str) -> "Query":
        """Parse the query string into a Query object.

        Args:
            query (str): The query string to parse.

        Returns:
            Query: The parsed query object.

        """
        if not cls.args_and_types:
            error_msg = "Query.args_and_types is not defined."
            raise ValueError(error_msg)

        # Check if the query string is valid
        rex = re.compile(r"(\w+)\((.*)\)")
        match = rex.search(query_str)
        if not match:
            error_msg = f"Invalid query syntax: {query_str}"
            raise ValueError(error_msg)

        # Check if the query name is valid
        query_name = match.group(1)
        if query_name not in Query.valid_queries:
            error_msg = f"Invalid query name: {query_name}"
            raise ValueError(error_msg)

        # Extract the query parameters
        rex = re.compile(r"(\w+)=(?:(\w+)|\[([^\]]+)\])")
        params = {}
        for arg, arg_val, arg_list in rex.findall(match.group(2)):
            if arg_val:
                params[arg] = arg_val
            elif arg_list:
                params[arg] = list(map(str.strip, arg_list.split(",")))
            else:
                error_msg = f"Invalid argument syntax: {(arg, arg_val, arg_list)}"
                raise ValueError(error_msg)
        return True
