import re
from typing import Any, ClassVar


class Query:
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

    def __init__(self, name: str, params: dict[str, Any]):
        self.name = name
        self.params = params

    @classmethod
    def parse(cls, query_str: str) -> "Query":
        """Parse the query string into a Query object.

        Args:
            query (str): The query string to parse.

        Returns:
            Query: The parsed query object.

        """
        return cls(name=query, params={})

    @staticmethod
    def check_valid(self, query_str: str) -> bool | None:
        """Check whether the query string is valid.

        The check includes both syntax and content validation.
        Raises an error if the query is invalid.

        Args:
            query_str (str): The query to check.

        Returns:
            bool: True if the query is valid.

        """
        rex = re.compile(r"(\w+)\((.*)\)")
        match = rex.match(query_str)
        if not match:
            error_msg = f"Invalid query syntax: {query_str}"
            raise SyntaxError(error_msg)

        query_name = match.group(1)
        if query_name not in self.valid_queries:
            error_msg = f"Invalid query name: {query_name}"
            raise ValueError(error_msg)

        query_params = [param.trim() for param in match.group(2).split(",")]
        return True
