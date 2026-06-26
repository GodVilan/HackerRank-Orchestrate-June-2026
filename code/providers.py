"""
providers.py — Thin provider abstraction over the model APIs.

Stage 1 (perception) and Stage 2 (reconciliation) call models only through this
layer, so the rest of the pipeline never touches an SDK directly. The Claude
provider wraps the Anthropic Messages API and centralizes:

  * the static SYSTEM block + Anthropic prompt caching (cache_control), so the
    long instructions are billed once across the ~82 perception calls;
  * single-image attachment in the user turn (base64 PNG from data_layer);
  * transient-error retry/backoff (tenacity), surfaced explicitly on top of the
    SDK's own retries;
  * adaptive temperature handling — see ClaudeProvider.generate.

Secrets are read from the environment via config.get_anthropic_api_key(); no key
is ever hardcoded, and constructing the provider does not require a key (the
client is built lazily on first call).
"""
from __future__ import annotations

import abc
import base64
from dataclasses import dataclass, field
from typing import Optional
from pydantic import BaseModel

import anthropic
import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

try:
    from google import genai
    from google.genai import errors as gemini_errors
    from google.genai import types as gemini_types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# Import config whether loaded as a package (`code.providers`) or top-level.
try:
    from . import config
except ImportError:  # pragma: no cover
    import config  # type: ignore


# ---------------------------------------------------------------------------
# Provider response
# ---------------------------------------------------------------------------
@dataclass
class ProviderResponse:
    """Normalized result of a single model call."""
    text: str
    model: str
    usage: dict = field(default_factory=dict)   # input/output/cache token counts
    stop_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Transient-error policy for tenacity
# ---------------------------------------------------------------------------
def _is_transient(exc: BaseException) -> bool:
    """Retry only on transient failures; never on 4xx (auth / bad request)."""
    if isinstance(exc, (
        anthropic.RateLimitError,            # 429
        anthropic.APIConnectionError,        # network / DNS / reset
        anthropic.APITimeoutError,           # read timeout
        anthropic.InternalServerError,       # 500
    )):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        # 529 overloaded and any other 5xx are retryable; 4xx are not.
        return getattr(exc, "status_code", 0) >= 500
    return False


_RETRY = retry(
    retry=retry_if_exception(_is_transient),
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)


def _is_gemini_transient(exc: BaseException) -> bool:
    """Retry on transient failures for Gemini (429, 5xx, or network issues)."""
    if HAS_GEMINI and isinstance(exc, gemini_errors.APIError):
        # 429: Rate limit, 500: Internal server error, etc.
        if exc.code in (429, 500, 503, 504, 529):
            return True
        if exc.code and exc.code >= 500:
            return True
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError)):
        return True
    return False


_GEMINI_RETRY = retry(
    retry=retry_if_exception(_is_gemini_transient),
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------
class LLMProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def generate(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_b64: Optional[str] = None,
        media_type: str = "image/png",
        images: Optional[list] = None,
        model: str,
        max_tokens: int,
        temperature: float,
        response_schema: Optional[type[BaseModel] | dict] = None,
    ) -> ProviderResponse:
        ...


# ---------------------------------------------------------------------------
# Claude provider
# ---------------------------------------------------------------------------
# Models that 400 on sampling params (discovered at runtime, memoized). The
# code does NOT pre-judge which models these are — it learns from a real 400 and
# never sends `temperature` to that model again. Any model that accepts
# temperature=0 therefore stays deterministic.
_NO_TEMPERATURE: set[str] = set()


class ClaudeProvider(LLMProvider):
    name = "anthropic"

    def __init__(self) -> None:
        self._client: Optional[anthropic.Anthropic] = None

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=config.get_anthropic_api_key())
        return self._client

    def _build_system(self, system_prompt: str) -> list[dict]:
        block: dict = {"type": "text", "text": system_prompt}
        if config.ENABLE_PROMPT_CACHING:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    @staticmethod
    def _image_block(b64: str, media_type: str) -> dict:
        return {"type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64}}

    def _build_user_content(
        self,
        user_text: str,
        image_b64: Optional[str],
        media_type: str,
        images: Optional[list] = None,
    ) -> list[dict]:
        content: list[dict] = []
        # Multi-image (baseline single-pass): list of {"data", "media_type"}.
        if images:
            for img in images:
                content.append(self._image_block(
                    img["data"], img.get("media_type", "image/png")))
        elif image_b64:
            content.append(self._image_block(image_b64, media_type))
        content.append({"type": "text", "text": user_text})
        return content

    @_RETRY
    def _call(self, **kwargs) -> "anthropic.types.Message":
        return self._get_client().messages.create(**kwargs)

    def generate(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_b64: Optional[str] = None,
        media_type: str = "image/png",
        images: Optional[list] = None,
        model: str,
        max_tokens: int,
        temperature: float,
        response_schema: Optional[type[BaseModel] | dict] = None,
    ) -> ProviderResponse:
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": self._build_system(system_prompt),
            "messages": [{
                "role": "user",
                "content": self._build_user_content(
                    user_text, image_b64, media_type, images),
            }],
        }
        if model not in _NO_TEMPERATURE:
            kwargs["temperature"] = temperature

        try:
            resp = self._call(**kwargs)
        except anthropic.BadRequestError as e:
            # Self-correct once if the model rejects the sampling parameter.
            msg = str(getattr(e, "message", "") or e).lower()
            if "temperature" in kwargs and "temperature" in msg:
                _NO_TEMPERATURE.add(model)
                kwargs.pop("temperature", None)
                resp = self._call(**kwargs)
            else:
                raise

        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        u = resp.usage
        usage = {
            "input_tokens": getattr(u, "input_tokens", 0),
            "output_tokens": getattr(u, "output_tokens", 0),
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        }
        return ProviderResponse(
            text=text, model=resp.model, usage=usage, stop_reason=resp.stop_reason
        )


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------
class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self) -> None:
        self._client: Optional[genai.Client] = None

    def _get_client(self) -> genai.Client:
        if not HAS_GEMINI:
            raise RuntimeError(
                "Gemini dependencies are not available. Ensure google-genai is installed."
            )
        if self._client is None:
            self._client = genai.Client(api_key=config.get_gemini_api_key())
        return self._client

    @_GEMINI_RETRY
    def _call(self, model: str, contents: list, config_obj: gemini_types.GenerateContentConfig) -> genai.types.GenerateContentResponse:
        return self._get_client().models.generate_content(
            model=model, contents=contents, config=config_obj
        )

    def generate(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_b64: Optional[str] = None,
        media_type: str = "image/png",
        images: Optional[list] = None,
        model: str,
        max_tokens: int,
        temperature: float,
        response_schema: Optional[type[BaseModel] | dict] = None,
    ) -> ProviderResponse:
        if not HAS_GEMINI:
            raise RuntimeError(
                "Gemini dependencies are not available. Ensure google-genai is installed."
            )

        # 1. Build contents list.
        contents: list = []
        import base64

        if images:
            for img in images:
                img_data = img["data"]
                img_media = img.get("media_type", "image/png")
                contents.append(
                    gemini_types.Part.from_bytes(
                        data=base64.b64decode(img_data),
                        mime_type=img_media
                    )
                )
        elif image_b64:
            contents.append(
                gemini_types.Part.from_bytes(
                    data=base64.b64decode(image_b64),
                    mime_type=media_type
                )
            )

        contents.append(user_text)

        # 2. Build configuration.
        config_args = {
            "system_instruction": system_prompt,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }

        if response_schema is not None:
            config_args["response_mime_type"] = "application/json"
            config_args["response_schema"] = response_schema

        config_obj = gemini_types.GenerateContentConfig(**config_args)

        # 3. Call the API.
        try:
            resp = self._call(model, contents, config_obj)
        except Exception as e:
            # Re-raise transient or API errors to be caught/handled
            raise e

        # Extract output text.
        text = resp.text or ""

        # Extract usage metrics.
        u = resp.usage_metadata
        usage = {
            "input_tokens": getattr(u, "prompt_token_count", 0) if u else 0,
            "output_tokens": getattr(u, "candidates_token_count", 0) if u else 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }

        return ProviderResponse(
            text=text,
            model=model,
            usage=usage,
            stop_reason=None
        )
