"""Define queries used in the Simulator with arguments and types."""

import logging
import re
from typing import Any, ClassVar, get_args, get_origin

from axs.config import Registerable
from axs.macroaction import MacroAction

logger = logging.getLogger(__name__)

QueryTypeMap = dict[str, dict[str, type]]

desc_arg_re = re.compile(r"<(\w+)>")
syntax_re = re.compile(r"(\w+)\((.*)\)")
args_re = re.compile(r"(\w+)=(?:(\w+)|\[([^\]]+)\]|\(([^\)]+)\))")


class Query(Registerable, class_type=None):
    """Represents queries with arguments used in Simulator.

    The name of the valid queries are fixed, but the arguments and types may
    be overriden to customize the query behavior. Currently the types supported are
    any classes that can parse a string and (non-nested) lists and tuples.
    """

    valid_queries: ClassVar[list[str]] = [
        "add",
        "remove",
        "whatif",
        "what",
    ]
    args_and_types: ClassVar[QueryTypeMap] = {
        "add": {"state": str},  # We don't know the state space so we use str
        "remove": {"agent": int},
        "whatif": {"agent": int, "actions": list[MacroAction], "time": int},
        "what": {"agent": int, "time": int},
    }

    def __init__(self, name: str, params: dict[str, Any]) -> "Query":
        """Initialize new Query.

        Args:
            name (str): The name of the query.
            params (dict): The parameters for the query.

        """
        self.quey_name = name
        self.params = params

    def __init_subclass__(cls) -> None:
        """Check validty of args_and_types."""
        super().__init_subclass__()
        if not cls.args_and_types:
            error_msg = "Query.args_and_types is not defined."
            raise ValueError(error_msg)

        for query_name in cls.valid_queries:
            for arg_name, arg_type in cls.args_and_types[query_name].items():
                origin = get_origin(arg_type)
                args = get_args(arg_type)
                if origin and any(arg is list or arg is tuple for arg in args):
                    error_msg = f"Invalid type {arg_type} for argument {arg_name}."
                    raise ValueError(error_msg)

        # Check description implementations
        query_descriptions = cls.query_descriptions()
        for query_name, description in query_descriptions.items():
            if query_name not in cls.valid_queries:
                error_msg = f"Invalid query name: {query_name}"
                raise ValueError(error_msg)
            for match in desc_arg_re.findall(description):
                if match not in cls.args_and_types[query_name]:
                    error_msg = (
                        f"Invalid query variable: {match} in description: {description}"
                    )
                    raise ValueError(error_msg)

    @classmethod
    def queries(cls) -> dict[str, str]:
        """Return the list of valid queries with arguments."""
        return {
            query: f"{query}({', '.join(args)})"
            for query, args in cls.args_and_types.items()
        }

    @classmethod
    def query_descriptions(cls) -> dict[str, str]:
        """Return a string with the query descriptions.

        If not overriden, then built-in descriptions are used.
        When overriden, you may refer to query variables in your description
        using the format <variable>. These descriptions are used to
        generate the user prompt for the LLM.
        """
        return {
            "add": "What would happen if a new agent was present from at <state> from the start?",  # noqa: E501
            "remove": "What would happen if <agent> was removed from the environment?",
            "whatif": "What would happen if <agent> took <actions> starting from <time>?",  # noqa: E501
            "what": "What will <agent> be doing at <time>?",
        }

    @classmethod
    def query_type_descriptions(cls) -> dict[str, str]:
        """Return a string with the query type descriptions.

        If not overriden, then descriptions are generated from the args_and_types
        automatically. The descriptions are used to generate user prompts for the LLM.
        """
        type_descriptions = {}
        for args in cls.args_and_types.values():
            for arg_name, arg_type in args.items():
                if arg_name in type_descriptions:
                    continue
                origin = get_origin(arg_type)
                if not origin:
                    type_descriptions[arg_name] = arg_type.__name__
                else:
                    args_names = ", ".join(arg.__name__ for arg in get_args(arg_type))
                    type_descriptions[arg_name] = f"{origin.__name__}[{args_names}]"
        return type_descriptions

    @classmethod
    def parse(cls, query_str: str) -> "Query":
        """Parse the query string into a Query object.

        Currently, only simple lists and tuples are supported arrays.
        Nested arrays are not supported.

        Args:
            query_str (str): The query string to parse.

        Returns:
            Query: The parsed query object.

        """
        # Check if the query string is valid
        match = syntax_re.search(query_str)
        if not match:
            error_msg = f"Invalid query syntax: {query_str}"
            raise ValueError(error_msg)

        # Check if the query name is valid
        query_name = match.group(1)
        if query_name not in Query.valid_queries:
            error_msg = f"Invalid query name: {query_name}"
            raise ValueError(error_msg)

        # Extract the query parameters
        params = {}
        for arg, arg_val, arg_list, arg_tuple in args_re.findall(match.group(2)):
            if arg_val:
                params[arg] = arg_val
            elif arg_list:
                params[arg] = list(map(str.strip, arg_list.split(",")))
            elif arg_tuple:
                params[arg] = list(map(str.strip, arg_tuple.split(",")))
            else:
                error_msg = f"Invalid argument syntax: {(arg, arg_val, arg_list)}"
                raise ValueError(error_msg)

        # Convert all parameters to the correct types
        arg_and_types = cls.args_and_types[query_name]
        for arg_name, arg_type in arg_and_types.items():
            if arg_name not in params:
                error_msg = f"Missing argument name: {arg_name}"
                raise ValueError(error_msg)
            params[arg_name] = cls._parse_type(arg_type, params[arg_name])

        return cls(query_name, params)

    @classmethod
    def _parse_type(cls, arg_type: type, params: str | list[str]) -> Any:
        """Parse the query parameter strings to Python types."""
        origin = get_origin(arg_type)
        args = get_args(arg_type)
        if not origin:
            return arg_type(params)

        if origin is tuple:
            values = []
            for arg in args:
                if arg is Ellipsis:
                    break
                values.append(arg(params.pop(0)))
            values.extend(params)
            return tuple(values)

        if origin is list:
            if isinstance(params, str):
                logger.warning("Converting raw string to list: %s", params)
            return [args[0](x) for x in params]
        error_msg = f"Invalid argument type: {arg_type}"
        raise ValueError(error_msg)
