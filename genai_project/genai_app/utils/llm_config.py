
# from langchain.llms import OpenAI

# def get_llama_maverick_llm():
#     return OpenAI(
#         model="meta-llama/llama-4-maverick-17b-128e-instruct",
#         openai_api_base="https://openrouter.ai/api/v1",
#         openai_api_key="YOUR_OPENROUTER_API_KEY"
#     )



# import requests
# import time
# import re
# from langchain.llms.base import LLM
# from typing import Optional, List, Mapping, Any
# from langchain.schema import Generation, LLMResult
# from loguru import logger

# import os

# GROQ_API_KEY = os.getenv("GROQ_API_KEY2")



# class GroqLLM(LLM):
#     model: str = "meta-llama/llama-4-maverick-17b-128e-instruct"
#     api_key: str = os.getenv("GROQ_API_KEY2") # Replace with env var in production
#     # api_key: str = os.getenv("GROQ_API_KEY") # Replace with env var in production
#    # Replace with env var in production
#     base_url: str = "https://api.groq.com/openai/v1"
#     temperature: float = 0.1
#     max_tokens: int = 1000

#     def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
#         system_prompt = kwargs.get("system_prompt", "You are a helpful SQL assistant.")
#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": prompt}
#         ]

#         payload = {
#             "model": self.model,
#             "messages": messages,
#             "temperature": self.temperature,
#             "max_tokens": self.max_tokens
#         }

#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }

#         max_retries = 3
#         for attempt in range(max_retries):
#             try:
#                 response = requests.post(
#                     f"{self.base_url}/chat/completions",
#                     headers=headers,
#                     json=payload,
#                     timeout=30
#                 )

#                 if response.status_code == 200:
#                     return response.json()["choices"][0]["message"]["content"].strip()

#                 elif response.status_code == 429:
#                     error_data = response.json()
#                     message = error_data.get("error", {}).get("message", "")
#                     logger.warning(f"Rate limit hit: {message}")
#                     match = re.search(r'try again in ([\\d\\.]+)s', message)
#                     wait_time = float(match.group(1)) if match else 2 ** attempt
#                     logger.info(f"Sleeping for {wait_time} seconds...")
#                     time.sleep(wait_time)
#                     continue

#                 else:
#                     logger.error(f"Groq API error: {response.status_code} - {response.text}")
#                     break

#             except Exception as e:
#                 logger.error(f"Groq API call failed: {e}")
#                 time.sleep(2 ** attempt)

#         return "I'm having trouble processing your request at the moment."

#     @property
#     def _llm_type(self) -> str:
#         return "groq-llama4"

#     def generate(self, prompts: List[str], stop: Optional[List[str]] = None,callbacks: Optional[Any] = None,  
#     **kwargs) -> LLMResult:
#         generations = [Generation(text=self._call(prompt, stop=stop)) for prompt in prompts]
#         return LLMResult(generations=[generations])

# def get_llama_maverick_llm():
#     return GroqLLM()




import os
import time
import logging
from typing import Any, List, Optional

from langchain.llms.base import LLM
from langchain.schema import LLMResult, Generation

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ServiceRequestError, ServiceResponseError

logger = logging.getLogger(__name__)


class AzureInferenceLLM(LLM):
    """LangChain LLM wrapper for Azure AI Inference Chat Completions."""

    # Core config
    endpoint: str = os.getenv("AZURE_INFERENCE_ENDPOINT", "")
    api_key: str = os.getenv("AZURE_INFERENCE_API_KEY", "")
    api_version: str = "2024-05-01-preview"
    model: str = os.getenv(
        "AZURE_INFERENCE_MODEL",
        "Llama-4-Maverick-17B-128E-Instruct-FP8-prochurn-demo",
    )

    # Generation params
    temperature: float = 0.1
    max_tokens: int = 1000
    top_p: float = 0.9
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    # Networking / retries
    request_timeout: int = 30
    max_retries: int = 3

    # lazy-initialized client cache
    _client_instance: Optional[ChatCompletionsClient] = None

    @property
    def _client(self) -> ChatCompletionsClient:
        if self._client_instance is None:
            if not self.endpoint or not self.api_key:
                raise ValueError(
                    "AzureInferenceLLM requires AZURE_INFERENCE_ENDPOINT and AZURE_INFERENCE_API_KEY."
                )
            self._client_instance = ChatCompletionsClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key),
                api_version=self.api_version,
            )
        return self._client_instance

    def _postprocess_with_stop(self, text: str, stop: Optional[List[str]]) -> str:
        if not stop or not text:
            return text
        # Trim on first occurrence of any stop token
        cut = len(text)
        for s in stop:
            idx = text.find(s)
            if idx != -1:
                cut = min(cut, idx)
        return text[:cut]

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        system_prompt = kwargs.get("system_prompt", "You are a helpful SQL assistant.")

        messages = [
            SystemMessage(content=system_prompt),
            UserMessage(content=prompt),
        ]

        params = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
            "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
            "model": kwargs.get("model", self.model),
        }

        # If the SDK supports stop (it does in recent previews), pass it through.
        if stop:
            params["stop"] = stop

        for attempt in range(self.max_retries):
            try:
                # azure-ai-inference handles timeouts via transport config; we keep our own retry loop
                response = self._client.complete(**params)  # type: ignore[arg-type]
                # Choices API: pick first, consistent with your current wrapper
                choice = response.choices[0]
                # Some SDKs expose .message.content; keep defensive fallback
                text = getattr(choice.message, "content", None) or getattr(choice, "text", "")
                text = (text or "").strip()

                # If stop not supported by server or not honored, apply client-side trim
                text = self._postprocess_with_stop(text, stop)
                return text

            except HttpResponseError as e:
                status = getattr(e, "status_code", None)
                body = getattr(e, "message", str(e))
                logger.warning(f"Azure Inference HTTP error (status={status}): {body}")
                if status == 429 and attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait = 2 ** attempt
                    logger.info(f"Rate limited. Backing off for {wait}s...")
                    time.sleep(wait)
                    continue
                break  # non-retryable or retries exhausted

            except (ServiceRequestError, ServiceResponseError, TimeoutError) as e:
                logger.warning(f"Azure Inference transient error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                break

            except Exception as e:
                logger.error(f"Azure Inference call failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                break

        return "I'm having trouble processing your request at the moment."

    @property
    def _llm_type(self) -> str:
        return "azure-ai-inference"

    def generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Any] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = [Generation(text=self._call(prompt, stop=stop, **kwargs)) for prompt in prompts]
        return LLMResult(generations=[generations])


def get_llama_maverick_llm() -> AzureInferenceLLM:
    """Factory to keep your existing call sites unchanged."""
    return AzureInferenceLLM()

# import os
# import time
# import logging
# from typing import Any, List, Optional

# from langchain.llms.base import LLM
# from langchain.schema import LLMResult, Generation

# from azure.ai.inference import ChatCompletionsClient
# from azure.ai.inference.models import SystemMessage, UserMessage
# from azure.core.credentials import AzureKeyCredential
# from azure.core.exceptions import HttpResponseError, ServiceRequestError, ServiceResponseError

# logger = logging.getLogger(__name__)


# class AzureInferenceLLM(LLM):
#     """LangChain LLM wrapper for Azure AI Inference Chat Completions."""

#     # Core config
#     endpoint: str = os.getenv("AZURE_INFERENCE_ENDPOINT", "")
#     api_key: str = os.getenv("AZURE_INFERENCE_API_KEY", "")
#     api_version: str = "2024-05-01-preview"
#     model: str = os.getenv(
#         "AZURE_INFERENCE_MODEL",
#         "Llama-4-Maverick-17B-128E-Instruct-FP8-prochurn-demo",
#     )

#     # Generation params
#     temperature: float = 0.1
#     max_tokens: int = 1000
#     top_p: float = 0.9
#     presence_penalty: float = 0.0
#     frequency_penalty: float = 0.0

#     # Networking / retries
#     request_timeout: int = 30
#     max_retries: int = 3

#     # lazy-initialized client cache
#     _client_instance: Optional[ChatCompletionsClient] = None

#     @property
#     def _client(self) -> ChatCompletionsClient:
#         if self._client_instance is None:
#             if not self.endpoint or not self.api_key:
#                 raise ValueError(
#                     "AzureInferenceLLM requires AZURE_INFERENCE_ENDPOINT and AZURE_INFERENCE_API_KEY."
#                 )
#             self._client_instance = ChatCompletionsClient(
#                 endpoint=self.endpoint,
#                 credential=AzureKeyCredential(self.api_key),
#                 api_version=self.api_version,
#             )
#         return self._client_instance

#     def _postprocess_with_stop(self, text: str, stop: Optional[List[str]]) -> str:
#         if not stop or not text:
#             return text
#         # Trim on first occurrence of any stop token
#         cut = len(text)
#         for s in stop:
#             idx = text.find(s)
#             if idx != -1:
#                 cut = min(cut, idx)
#         return text[:cut]

#     def _call(
#         self,
#         prompt: str,
#         stop: Optional[List[str]] = None,
#         **kwargs: Any,
#     ) -> str:
#         system_prompt = kwargs.get("system_prompt", "You are a helpful SQL assistant.")

#         messages = [
#             SystemMessage(content=system_prompt),
#             UserMessage(content=prompt),
#         ]

#         params = {
#             "messages": messages,
#             "max_tokens": kwargs.get("max_tokens", self.max_tokens),
#             "temperature": kwargs.get("temperature", self.temperature),
#             "top_p": kwargs.get("top_p", self.top_p),
#             "presence_penalty": kwargs.get("presence_penalty", self.presence_penalty),
#             "frequency_penalty": kwargs.get("frequency_penalty", self.frequency_penalty),
#             "model": kwargs.get("model", self.model),
#         }

#         # If the SDK supports stop (it does in recent previews), pass it through.
#         if stop:
#             params["stop"] = stop

#         for attempt in range(self.max_retries):
#             try:
#                 # azure-ai-inference handles timeouts via transport config; we keep our own retry loop
#                 response = self._client.complete(**params)  # type: ignore[arg-type]
#                 # Choices API: pick first, consistent with your current wrapper
#                 choice = response.choices[0]
#                 # Some SDKs expose .message.content; keep defensive fallback
#                 text = getattr(choice.message, "content", None) or getattr(choice, "text", "")
#                 text = (text or "").strip()

#                 # If stop not supported by server or not honored, apply client-side trim
#                 text = self._postprocess_with_stop(text, stop)
#                 return text

#             except HttpResponseError as e:
#                 status = getattr(e, "status_code", None)
#                 body = getattr(e, "message", str(e))
#                 logger.warning(f"Azure Inference HTTP error (status={status}): {body}")
#                 if status == 429 and attempt < self.max_retries - 1:
#                     # Exponential backoff
#                     wait = 2 ** attempt
#                     logger.info(f"Rate limited. Backing off for {wait}s...")
#                     time.sleep(wait)
#                     continue
#                 break  # non-retryable or retries exhausted

#             except (ServiceRequestError, ServiceResponseError, TimeoutError) as e:
#                 logger.warning(f"Azure Inference transient error: {e}")
#                 if attempt < self.max_retries - 1:
#                     time.sleep(2 ** attempt)
#                     continue
#                 break

#             except Exception as e:
#                 logger.error(f"Azure Inference call failed: {e}")
#                 if attempt < self.max_retries - 1:
#                     time.sleep(2 ** attempt)
#                     continue
#                 break

#         return "I'm having trouble processing your request at the moment."

#     @property
#     def _llm_type(self) -> str:
#         return "azure-ai-inference"

#     def generate(
#         self,
#         prompts: List[str],
#         stop: Optional[List[str]] = None,
#         callbacks: Optional[Any] = None,
#         **kwargs: Any,
#     ) -> LLMResult:
#         generations = [Generation(text=self._call(prompt, stop=stop, **kwargs)) for prompt in prompts]
#         return LLMResult(generations=[generations])


# def get_llama_maverick_llm() -> AzureInferenceLLM:
#     """Factory to keep your existing call sites unchanged."""
#     return AzureInferenceLLM()
