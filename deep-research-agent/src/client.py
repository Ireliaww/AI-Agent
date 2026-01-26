"""
Gemini Client Module - Async wrapper for Google's new genai SDK.

This module provides an async interface to Google's Gemini models using
the official google-genai SDK (NOT the deprecated google-generativeai).
"""

import os
from typing import Optional, AsyncIterator, Any
from dataclasses import dataclass

from google import genai
from google.genai import types


# System instruction for the Research Agent
RESEARCH_AGENT_SYSTEM_INSTRUCTION = """You are a Deep Research Agent powered by Gemini.

Your role is to help users conduct thorough research on any topic by:
1. Breaking down complex questions into searchable queries
2. Analyzing search results to extract relevant information
3. Synthesizing findings into comprehensive, well-structured reports

Guidelines:
- Always cite your sources when presenting information
- Be objective and present multiple perspectives when relevant
- Clearly distinguish between facts and opinions
- Acknowledge limitations or gaps in available information
- Use clear, professional language suitable for research reports

When generating search queries:
- Create diverse queries that cover different aspects of the topic
- Include both broad and specific queries
- Consider related concepts and synonyms

When analyzing information:
- Focus on credible sources
- Look for consensus and contradictions
- Identify key facts, figures, and expert opinions
"""


@dataclass
class ModelConfig:
    """Configuration for the Gemini model."""

    model_name: str = "gemini-2.5-flash"
    fallback_models: tuple = ("gemini-2.0-flash", "gemini-1.5-flash")
    temperature: float = 0.7
    max_output_tokens: int = 8192


class GeminiClient:
    """
    Async wrapper for Google's Gemini API using the new google-genai SDK.

    This client handles:
    - API key management
    - Model fallback (if primary model is unavailable)
    - Async content generation
    - Tool/function calling support
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[ModelConfig] = None
    ):
        """
        Initialize the Gemini client.

        Args:
            api_key: Google API key. If not provided, reads from GOOGLE_API_KEY env var.
            config: Model configuration. Uses defaults if not provided.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.config = config or ModelConfig()
        self._client: Optional[genai.Client] = None
        self._active_model: Optional[str] = None

    @property
    def client(self) -> genai.Client:
        """Lazy initialization of the genai client."""
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @property
    def active_model(self) -> str:
        """Returns the currently active model name."""
        return self._active_model or self.config.model_name

    async def _try_model(self, model_name: str, **kwargs) -> Optional[Any]:
        """
        Attempt to use a specific model.

        Args:
            model_name: The model to try.
            **kwargs: Arguments to pass to generate_content.

        Returns:
            Response if successful, None if model is unavailable.
        """
        try:
            response = await self.client.aio.models.generate_content(
                model=model_name,
                **kwargs
            )
            self._active_model = model_name
            return response
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg or "not supported" in error_msg:
                return None
            raise

    async def generate_content(
        self,
        contents: str | list,
        tools: Optional[list] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> Any:
        """
        Generate content using Gemini with automatic model fallback.

        Args:
            contents: The input content (prompt or conversation).
            tools: Optional list of tools/functions for the model to use.
            system_instruction: Override the default system instruction.
            temperature: Override the default temperature.
            max_output_tokens: Override the default max output tokens.

        Returns:
            The model's response.

        Raises:
            RuntimeError: If all models fail.
        """
        # Build configuration
        config_kwargs = {
            "temperature": temperature or self.config.temperature,
            "max_output_tokens": max_output_tokens or self.config.max_output_tokens,
        }

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        else:
            config_kwargs["system_instruction"] = RESEARCH_AGENT_SYSTEM_INSTRUCTION

        if tools:
            config_kwargs["tools"] = tools

        generation_config = types.GenerateContentConfig(**config_kwargs)

        # Prepare kwargs for generate_content
        kwargs = {
            "contents": contents,
            "config": generation_config,
        }

        # Try primary model first
        models_to_try = [self.config.model_name] + list(self.config.fallback_models)

        last_error = None
        for model_name in models_to_try:
            try:
                response = await self._try_model(model_name, **kwargs)
                if response is not None:
                    return response
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(
            f"All models failed. Tried: {models_to_try}. "
            f"Last error: {last_error}"
        )

    async def generate_with_tools(
        self,
        contents: str | list,
        tools: list,
        system_instruction: Optional[str] = None,
    ) -> Any:
        """
        Generate content with tool/function calling support.

        This is a convenience method for tool-augmented generation.

        Args:
            contents: The input content.
            tools: List of tools available to the model.
            system_instruction: Optional system instruction override.

        Returns:
            The model's response (may include tool calls).
        """
        return await self.generate_content(
            contents=contents,
            tools=tools,
            system_instruction=system_instruction,
        )

    async def chat(
        self,
        messages: list[dict],
        tools: Optional[list] = None,
        system_instruction: Optional[str] = None,
    ) -> Any:
        """
        Multi-turn conversation support.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            tools: Optional tools for the model.
            system_instruction: Optional system instruction override.

        Returns:
            The model's response.
        """
        # Convert messages to the format expected by genai
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Map roles to genai format
            if role == "assistant":
                role = "model"

            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=content)]
                )
            )

        return await self.generate_content(
            contents=contents,
            tools=tools,
            system_instruction=system_instruction,
        )


# Convenience function for quick usage
async def quick_generate(prompt: str, api_key: Optional[str] = None) -> str:
    """
    Quick helper for simple generation without full client setup.

    Args:
        prompt: The prompt to send.
        api_key: Optional API key.

    Returns:
        The generated text.
    """
    client = GeminiClient(api_key=api_key)
    response = await client.generate_content(prompt)
    return response.text


if __name__ == "__main__":
    # Simple test
    import asyncio

    async def test():
        client = GeminiClient()
        response = await client.generate_content("Hello, who are you?")
        print(f"Model: {client.active_model}")
        print(f"Response: {response.text}")

    asyncio.run(test())
