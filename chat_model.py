from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type
import os

import anthropic
from openai import OpenAI
from pydantic import BaseModel
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig
from mistralai import Mistral

from config import OPENAI_API_KEY as CONFIG_OPENAI_API_KEY, MODEL, PROVIDER, \
    CF_ACCOUNT_ID as CONFIG_CF_ACCOUNT_ID, CF_API_KEY as CONFIG_CF_API_KEY, \
    CLAUDE_API_KEY as CONFIG_CLAUDE_API_KEY, GEMINI_API_KEY as CONFIG_GEMINI_API_KEY, \
    MISTRAL_API_KEY as CONFIG_MISTRAL_API_KEY


# ============================================================================
# API Keys: Environment variables with fallback to config.py
# ============================================================================

def get_api_key(env_var: str, config_fallback: str) -> str:
    """Get API key from environment variable, falling back to config.py value."""
    return os.environ.get(env_var, config_fallback)


# API keys with environment variable priority
OPENAI_API_KEY = get_api_key("OPENAI_API_KEY", CONFIG_OPENAI_API_KEY)
GEMINI_API_KEY = get_api_key("GEMINI_API_KEY", CONFIG_GEMINI_API_KEY)
MISTRAL_API_KEY = get_api_key("MISTRAL_API_KEY", CONFIG_MISTRAL_API_KEY)
CLAUDE_API_KEY = get_api_key("CLAUDE_API_KEY", CONFIG_CLAUDE_API_KEY)
CF_API_KEY = get_api_key("CF_API_KEY", CONFIG_CF_API_KEY)
CF_ACCOUNT_ID = get_api_key("CF_ACCOUNT_ID", CONFIG_CF_ACCOUNT_ID)


class ChatModel(ABC):
    """Abstract base class for chat models."""

    @abstractmethod
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        pass

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None, # Can be OpenAI dict or Pydantic model for Ollama
    ) -> str:
        """Generate a response based on the message history."""
        pass

    @abstractmethod
    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[BaseModel],
        temperature: float = 0.0,
        top_p: Optional[float] = None,
    ) -> BaseModel:
        """Generate a structured response conforming to the Pydantic model."""
        pass


class OpenAIChatModel(ChatModel):
    """Chat model implementation using the OpenAI API."""

    def __init__(self, model_name: str, api_key: Optional[str] = OPENAI_API_KEY):
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        # Token accounting
        self.last_usage: Optional[Dict[str, int]] = None
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_total_tokens: int = 0

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1**50,
        top_p: float = 1e-9,
        response_format: Optional[Dict[str, Any]] | Type[BaseModel] = None,
    ) -> str:
        """Generate a text response using OpenAI."""
        completion_args = {
            "model": self.model_name,
            "instructions": messages[0]["content"],
            "input": messages[1]["content"],
            "temperature": temperature,
        }
        completion_args["top_p"] = top_p
        # if response_format is not None and \
        #         isinstance(response_format, type) and issubclass(response_format, BaseModel):
        #     # If it's a Pydantic model, we can use it for validation after getting the response
        #     completion_args["response_format"] = response_format
        #
        #     completion = self.client.beta.chat.completions.parse(**completion_args)
        # else:
        if "5" in self.model_name:
            completion = self.client.responses.create(model=self.model_name,
                                                      instructions=messages[0]["content"],
                                                      input=messages[-1]["content"],
                                                      reasoning={
                                                          "effort": "low"
                                                      }
                                                      )
            return completion.output[1].content[0].text

        else:
            completion = self.client.responses.create(model=self.model_name,
                                                      instructions=messages[0]["content"],
                                                      input=messages[-1]["content"],
                                                      temperature=temperature)
        try:
            usage = getattr(completion, "usage", None)
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            if usage is not None:
                # Handle SDK objects and dicts defensively
                if hasattr(usage, "prompt_tokens") or hasattr(usage, "completion_tokens") or hasattr(usage, "total_tokens"):
                    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                    # Some SDKs include total_tokens; otherwise derive it
                    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens))
                elif isinstance(usage, dict):
                    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
                    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
                    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
            self.last_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_total_tokens += total_tokens
        except Exception:
            # Best-effort: do not fail the call if usage is unavailable
            self.last_usage = None
        return completion.output[0].content[0].text

    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[BaseModel],
        temperature: float = 0.0,
        top_p: float = 1e-9,
    ) -> BaseModel:
        """Generate a structured response using OpenAI's function calling/structured output."""
        # OpenAI's newer approach often uses the `response_format` with JSON,
        # or for more complex Pydantic models, one might use tool calls.
        # Here, we'll use the JSON mode and Pydantic validation.
        
        # Request JSON output
        response_content = self.generate(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            response_format=response_model
        )
        
        # Validate the JSON output against the Pydantic model
        try:
            # Assuming the response_content is a valid JSON string
            validated_data = response_model.model_validate_json(response_content)
            return validated_data
        except Exception as e:
             print(f"Error validating OpenAI JSON response against {response_model.__name__}: {e}")
             print(f"Received content: {response_content}")
             # Handle error: maybe retry, return a default, or raise
             raise ValueError(f"Could not parse OpenAI response into {response_model.__name__}.") from e

    # Note: OpenAI's client.beta.chat.completions.parse is deprecated/removed.
    # The modern way is to request JSON and validate, or use tool calling for Pydantic models.


class CloudflareChatModel(ChatModel):
    """Chat model implementation using Ollama."""
    def __init__(
        self,
        model_name: str,                               # e.g. "@cf/meta/llama-3.1-8b-instruct"
        api_key: str,                                  # Cloudflare API token with AI:Run permission
        base_url: Optional[str] = None,                # override if using AI Gateway
    ):
        if base_url is None:
            # Workers AI OpenAI-compatible base URL
            base_url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1"

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.last_usage: Optional[Dict[str, int]] = None
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_total_tokens: int = 0


    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        response_format: Optional[str] = None,         # "json" for JSON object mode
    ) -> str:
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4096,  # Adjust as needed based on model limits
        }
        if top_p is not None:
            kwargs["top_p"] = top_p

        # OpenAI-compatible JSON mode
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}  # structured JSON mode

        resp = self.client.chat.completions.create(**kwargs)
        try:
            usage = getattr(resp, "usage", None)
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            if usage is not None:
                # Handle SDK objects and dicts defensively
                if hasattr(usage, "prompt_tokens") or hasattr(usage, "completion_tokens") or hasattr(usage, "total_tokens"):
                    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                    # Some SDKs include total_tokens; otherwise derive it
                    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens))
                elif isinstance(usage, dict):
                    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
                    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
                    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
            self.last_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_total_tokens += total_tokens
        except Exception:
            # Best-effort: do not fail the call if usage is unavailable
            self.last_usage = None

        return resp.choices[0].message.content

    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[BaseModel],
        temperature: float = 0.0,
        top_p: Optional[float] = None,
    ) -> BaseModel:
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if top_p is not None:
            kwargs["top_p"] = top_p

        # Full JSON Schema mode (Workers AI supports OpenAI-style JSON mode)
        # If you prefer a lighter touch, you can still use {"type": "json_object"} as above.
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": response_model.model_json_schema()
        }

        resp = self.client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content

        # Validate with Pydantic v2
        try:
            return response_model.model_validate_json(content)
        except Exception as e:
            # Fallback: some models wrap JSON in code fences—strip if needed
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                # remove optional language hint like ```json
                cleaned = "\n".join(line for line in cleaned.splitlines() if not line.lower().startswith("json")).strip()
            return response_model.model_validate_json(cleaned)

class AnthropicChatModel(ChatModel):
    """Chat model implementation using the OpenAI API."""

    def __init__(self, model_name: str, api_key: Optional[str] = OPENAI_API_KEY):
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name

        # Token accounting
        self.last_usage: Optional[Dict[str, int]] = None
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_total_tokens: int = 0

    def generate(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.1 ** 50,
            top_p: float = 1e-9,
            response_format: Optional[Dict[str, Any]] | Type[BaseModel] = None,
    ) -> str:
        completion = self.client.messages.create(
            model=self.model_name,
            max_tokens=12400,
            system=messages[0]["content"],
            messages=messages[1:],
            temperature=temperature
        )
        try:
            usage = getattr(completion, "usage", None)
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            if usage is not None:
                # Handle SDK objects and dicts defensively
                if hasattr(usage, "input_tokens") or hasattr(usage, "output_tokens"):
                    prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0)
                    completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
                    # Some SDKs include total_tokens; otherwise derive it
                    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or (
                                prompt_tokens + completion_tokens))
                elif isinstance(usage, dict):
                    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
                    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
                    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
            self.last_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_total_tokens += total_tokens
        except Exception:
            # Best-effort: do not fail the call if usage is unavailable
            self.last_usage = None
        return completion.content[0].text

    def generate_structured(
            self,
            messages: List[Dict[str, str]],
            response_model: Type[BaseModel],
            temperature: float = 0.0,
            top_p: float = 1e-9,
    ) -> BaseModel:
        """Generate a structured response using OpenAI's function calling/structured output."""
        # OpenAI's newer approach often uses the `response_format` with JSON,
        # or for more complex Pydantic models, one might use tool calls.
        # Here, we'll use the JSON mode and Pydantic validation.

        # Request JSON output
        response_content = self.generate(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            response_format=response_model
        )

        # Validate the JSON output against the Pydantic model
        try:
            # Assuming the response_content is a valid JSON string
            validated_data = response_model.model_validate_json(response_content)
            return validated_data
        except Exception as e:
            print(f"Error validating OpenAI JSON response against {response_model.__name__}: {e}")
            print(f"Received content: {response_content}")
            # Handle error: maybe retry, return a default, or raise
            raise ValueError(f"Could not parse OpenAI response into {response_model.__name__}.") from e

    # Note: OpenAI's client.beta.chat.completions.parse is deprecated/removed.
    # The modern way is to request JSON and validate, or use tool calling for Pydantic models.

class GoogleChatModel(ChatModel):
    """Chat model implementation using the OpenAI API."""

    def __init__(self, model_name: str, api_key: Optional[str] = OPENAI_API_KEY):
        if not api_key:
            raise ValueError("Gemini API key is required.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        # Token accounting
        self.last_usage: Optional[Dict[str, int]] = None
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_total_tokens: int = 0

    def generate(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.1 ** 50,
            top_p: float = 1e-9,
            response_format: Optional[Dict[str, Any]] | Type[BaseModel] = None,
    ) -> str:
        """Generate a text response using Google genai API."""

        if "5" in self.model_name:
            completion = self.client.models.generate_content(
                model=self.model_name,
                contents=messages[1]["content"],
                config=GenerateContentConfig(
                    system_instruction=messages[0]["content"],
                    temperature=temperature,
                    thinking_config=ThinkingConfig(thinking_budget=1024)
                )
            )
        else:
            completion = self.client.models.generate_content(
                model=self.model_name,
                contents=messages[1]["content"],
                config=GenerateContentConfig(
                    system_instruction=messages[0]["content"],
                    temperature=temperature
                )
            )
        try:
            usage = getattr(completion, "usage_metadata", None)
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            if usage is not None:
                # Handle SDK objects and dicts defensively
                if hasattr(usage, "prompt_token_count") or hasattr(usage, "candidates_token_count") or hasattr(usage,
                                                                                                     "total_token_count"):
                    prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
                    completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
                    # Some SDKs include total_tokens; otherwise derive it
                    total_tokens = int(getattr(usage, "total_token_count", prompt_tokens + completion_tokens) or (
                                prompt_tokens + completion_tokens))
                elif isinstance(usage, dict):
                    prompt_tokens = int(usage.get("prompt_token_count") or usage.get("prompt_token_count") or 0)
                    completion_tokens = int(usage.get("candidates_token_count") or usage.get("candidates_token_count") or 0)
                    total_tokens = int(usage.get("total_token_count") or (prompt_tokens + completion_tokens))
            self.last_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_total_tokens += total_tokens
        except Exception:
            # Best-effort: do not fail the call if usage is unavailable
            self.last_usage = None
        return completion.candidates[0].content.parts[0].text

    def generate_structured(
            self,
            messages: List[Dict[str, str]],
            response_model: Type[BaseModel],
            temperature: float = 0.0,
            top_p: float = 1e-9,
    ) -> BaseModel:
        """Generate a structured response using OpenAI's function calling/structured output."""
        # OpenAI's newer approach often uses the `response_format` with JSON,
        # or for more complex Pydantic models, one might use tool calls.
        # Here, we'll use the JSON mode and Pydantic validation.

        # Request JSON output
        response_content = self.generate(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            response_format=response_model
        )

        # Validate the JSON output against the Pydantic model
        try:
            # Assuming the response_content is a valid JSON string
            validated_data = response_model.model_validate_json(response_content)
            return validated_data
        except Exception as e:
            print(f"Error validating OpenAI JSON response against {response_model.__name__}: {e}")
            print(f"Received content: {response_content}")
            # Handle error: maybe retry, return a default, or raise
            raise ValueError(f"Could not parse OpenAI response into {response_model.__name__}.") from e

    # Note: OpenAI's client.beta.chat.completions.parse is deprecated/removed.
    # The modern way is to request JSON and validate, or use tool calling for Pydantic models.
class LeChatModel(ChatModel):
    """Chat model implementation using the OpenAI API."""

    def __init__(self, model_name: str, api_key: Optional[str] = OPENAI_API_KEY):
        if not api_key:
            raise ValueError("Gemini API key is required.")
        self.client = Mistral(api_key=api_key)
        self.model_name = model_name
        # Token accounting
        self.last_usage: Optional[Dict[str, int]] = None
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_total_tokens: int = 0

    def generate(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.1 ** 50,
            top_p: float = 1e-9,
            response_format: Optional[Dict[str, Any]] | Type[BaseModel] = None,
    ) -> str:
        """Generate a text response using Google genai API."""

        contents = [{
          "role": "system",
          "content": [
            {
              "type": "text",
              "text": messages[0]["content"]
            }
          ]
        },{
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": messages[1]["content"]
            }
          ]
        }]

        completion = self.client.chat.complete(
                model=self.model_name,
                messages=contents,
                temperature=temperature,
                max_tokens=8192
        )
        self.last_usage = None
        return completion.choices[0].message.content[1].text

    def generate_structured(
            self,
            messages: List[Dict[str, str]],
            response_model: Type[BaseModel],
            temperature: float = 0.0,
            top_p: float = 1e-9,
    ) -> BaseModel:
        """Generate a structured response using OpenAI's function calling/structured output."""
        # OpenAI's newer approach often uses the `response_format` with JSON,
        # or for more complex Pydantic models, one might use tool calls.
        # Here, we'll use the JSON mode and Pydantic validation.

        # Request JSON output
        response_content = self.generate(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            response_format=response_model
        )

        # Validate the JSON output against the Pydantic model
        try:
            # Assuming the response_content is a valid JSON string
            validated_data = response_model.model_validate_json(response_content)
            return validated_data
        except Exception as e:
            print(f"Error validating OpenAI JSON response against {response_model.__name__}: {e}")
            print(f"Received content: {response_content}")
            # Handle error: maybe retry, return a default, or raise
            raise ValueError(f"Could not parse OpenAI response into {response_model.__name__}.") from e

    # Note: OpenAI's client.beta.chat.completions.parse is deprecated/removed.
    # The modern way is to request JSON and validate, or use tool calling for Pydantic models.


def get_chat_model(model_provider: Optional[str] = None,
        model_name: Optional[str] = None) -> ChatModel:
    """Factory function to get the configured chat model."""
    if model_provider is None:
        model_provider = PROVIDER
    if model_name is None:
        model_name = MODEL

    if model_provider.lower() == "openai":
        return OpenAIChatModel(model_name=model_name, api_key=OPENAI_API_KEY)
    elif model_provider.lower() == "cloudflare":
        # Ensure the model name format is suitable for Ollama (e.g., 'llama3.1:8b')
        return CloudflareChatModel(model_name=model_name, api_key=CF_API_KEY)
    elif model_provider.lower() == "google":
        return GoogleChatModel(model_name=model_name, api_key=GEMINI_API_KEY)
    elif model_provider.lower() == "anthropic":
        return AnthropicChatModel(model_name=model_name, api_key=CLAUDE_API_KEY)
    elif model_provider.lower() == "mistral":
        return LeChatModel(model_name=model_name, api_key=MISTRAL_API_KEY)
    else:
        raise ValueError(f"Unsupported chat model provider: {model_provider}")
