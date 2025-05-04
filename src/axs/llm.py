"""Provides a convenience class for using various LLM models.

Supports offline, localhost, and online LLM models.
"""

import importlib
import logging
import os

import anthropic
import openai

from axs.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMWrapper:
    """A convenience class for using locally hosted, offline, or online LLM models."""

    def __init__(self, config: LLMConfig) -> "LLMWrapper":
        """Initialize the LLMWrapper with the configuration.

        Args:
            config (LLMConfig): The configuration object for the LLMWrapper.

        """
        self.config = config

        self._llm = None
        if "stop" not in config.sampling_params:
            config.sampling_params["stop"] = ["<|endoftext|>"]
        self._sampling_params = config.sampling_params
        self._mode = config.host_location

        if self._mode == "offline":
            from vllm import LLM, SamplingParams

            self._sampling_params = SamplingParams(**self._sampling_params)
            self._llm = LLM(
                config.model,
                seed=self._sampling_params.seed,
                **config.model_kwargs,
            )
        elif self._mode == "localhost":
            base_url = config.base_url
            if not base_url:
                base_url = "http://localhost:8000/v1"
            self._llm = openai.OpenAI(api_key="EMPTY", base_url=base_url)
        elif self._mode == "online":
            if not config.base_url or "OPENAI" in config.api_key_env_var:
                self._llm = openai.OpenAI()
            elif "ANTHROPIC" in config.api_key_env_var:
                self._llm = anthropic.Anthropic()
            else:
                api_key = os.getenv(config.api_key_env_var)
                self._llm = openai.OpenAI(
                    base_url=config.base_url,
                    api_key=api_key,
                )
        else:
            error_msg = f"Invalid inference mode: {self._mode}"
            raise ValueError(error_msg)

    def chat(self, messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict]:
        """Chat with the LLM model using the given messages.

        Args:
            messages (List[Dict[str, str]]): List of messages to send to the LLM model.

        """
        vllm_installed = importlib.util.find_spec("vllm") is not None
        if self._mode == "offline" and not vllm_installed:
            error_msg = "vLLM not install but offline sampling was given."
            raise ValueError(error_msg)

        if isinstance(self._sampling_params, dict):
            if vllm_installed:
                from vllm import SamplingParams

                self._sampling_params = SamplingParams(**self._sampling_params)
            else:
                from axs.config import SamplingParams

                self._sampling_params = SamplingParams(self._sampling_params)

        _messages = self.merge_consecutive_messages(messages)

        if self._mode == "offline":
            outputs = self._llm.chat(_messages, sampling_params=self._sampling_params)
            return [
                {"role": "assistant", "content": ot.outputs[0].text} for ot in outputs
            ]

        available_models = [m.id for m in self._llm.models.list().data]
        internal_name = [
            model for model in available_models if self.config.model in model
        ]
        if not internal_name:
            error_msg = (
                f"Model {self.config.model} not found in openai.OpenAI "
                f"available models: {available_models}"
            )
            raise ValueError(error_msg)
        if len(internal_name) > 1:
            error_msg = (
                f"Multiple models found for {self.config.model}: {internal_name}"
            )
            raise ValueError(error_msg)

        # TODO: Add support for the responses OpenAI API
        if isinstance(self._llm, anthropic.Anthropic):
            system_prompt = messages[0]
            if system_prompt["role"] != "system":
                error_msg = "First message must be system prompt."
                raise ValueError(error_msg)

            message = self._llm.messages.create(
                model=internal_name[0],
                messages=messages[1:],  # Skip the system prompt
                stream=False,
                max_tokens=self._sampling_params.max_tokens,
                stop_sequences=self._sampling_params.stop,
                temperature=self._sampling_params.temperature,
                top_p=self._sampling_params.top_p,
                system=system_prompt["content"],
            )
            responses = [
                {"role": message.role, "content": m.text} for m in message.content
            ]
            usage = {
                "prompt_tokens": message.usage.input_tokens,
                "completion_tokens": message.usage.output_tokens,
                "total_tokens": message.usage.input_tokens
                + message.usage.output_tokens
                + message.usage.cache_creation_input_tokens
                + message.usage.cache_read_input_tokens,
            }
        elif "o4-mini" in internal_name[0]:
            system_prompt = messages[0]
            if system_prompt["role"] != "system":
                error_msg = "First message must be system prompt."
                raise ValueError(error_msg)

            system_prompt_content = system_prompt["content"]
            system_prompt_content = system_prompt_content.replace(
                "\nEnd all of your responses with '<|endoftext|>'.", "",
            )
            response = self._llm.responses.create(
                model="o4-mini",
                instructions=system_prompt_content,
                reasoning={"effort": "medium"},
                input=messages[1:],
                max_output_tokens=16384,
            )

            if (
                response.status == "incomplete"
                and response.incomplete_details.reason == "max_output_tokens"
            ):
                logger.warning("Ran out of tokens")
                if response.output_text:
                    logger.warning("Partial output: %s", response.output_text)
                else:
                    logger.warning("Ran out of tokens during reasoning")

            responses = [self.wrap("assistant", response.output_text)]
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens
                + response.usage.output_tokens,
            }

        else:
            completions = self._llm.chat.completions.create(
                model=internal_name[0],
                messages=_messages,
                stream=False,
                seed=self._sampling_params.seed,
                n=self._sampling_params.n,
                logit_bias=self._sampling_params.logit_bias,
                logprobs=self._sampling_params.logprobs,
                presence_penalty=self._sampling_params.presence_penalty,
                frequency_penalty=self._sampling_params.frequency_penalty,
                stop=self._sampling_params.stop,
                max_tokens=self._sampling_params.max_tokens,
                temperature=self._sampling_params.temperature,
                top_p=self._sampling_params.top_p,
            )
            responses = [
                {"role": c.message.role, "content": c.message.content.strip()}
                for c in completions.choices
            ]
            usage = {
                "prompt_tokens": completions.usage.prompt_tokens,
                "completion_tokens": completions.usage.completion_tokens,
                "total_tokens": completions.usage.total_tokens,
            }

        logger.info(
            "[bold yellow]LLM response:[/bold yellow]\n%s",
            responses[0]["content"],
            extra={"markup": True},
        )

        return responses, usage

    @property
    def mode(self) -> str:
        """Get the current inference mode."""
        return self._mode

    @staticmethod
    def merge_consecutive_messages(
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Merge consecutive messages with the same role.

        Args:
            messages (List[Dict[str, str]]): List of messages to merge.

        """
        merged_messages = []
        for message in messages:
            if not merged_messages or message["role"] != merged_messages[-1]["role"]:
                merged_messages.append(message)
            else:
                merged_messages[-1]["content"] += "\n\n" + message["content"]
        return merged_messages

    @staticmethod
    def wrap(
        role: str,
        content: str,
        **kwargs: dict[str, str],
    ) -> dict[str, str]:
        """Wrap the message for submission to an OpenAI style chat API.

        Args:
            role (str): The role of the message sender.
            content (str): The content of the message.
            kwargs: Additional keyword arguments.

        """
        return {"role": role, "content": content, **kwargs}
