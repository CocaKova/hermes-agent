"""Thinking-tolerant Graphiti LLM client for SILAS's local brain.

The 35B is served by vLLM with a qwen3 reasoning parser. For Graphiti's
extraction calls it routes the entire (valid JSON) answer into the response's
``reasoning`` field and leaves ``content`` null — so Graphiti's default client
sees an empty body and raises EmptyResponseError, breaking entity/edge
extraction (and thus all auto-capture).

This subclass overrides ``_generate_response`` to:
  1. ask supported local models to skip thinking/reasoning for extraction;
  2. fall back to the ``reasoning`` field when ``content`` is empty — that's
     where the JSON actually is on this stack.

Everything else (schema response_format, fence stripping, parsing) is unchanged.
"""

from __future__ import annotations

import json
import logging
import typing

import openai

from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client.errors import EmptyResponseError, RateLimitError

logger = logging.getLogger(__name__)


def _extract_body(message) -> str:
    """Prefer content; fall back to the vLLM reasoning channel."""
    body = (getattr(message, "content", None) or "").strip()
    if body:
        return body
    reasoning = getattr(message, "reasoning", None)
    if not reasoning:
        extra = getattr(message, "model_extra", None) or {}
        reasoning = extra.get("reasoning")
    return (reasoning or "").strip()


def _request_extras_for_model(model: str) -> dict[str, typing.Any]:
    """Return model-specific extraction request knobs.

    Mistral's native vLLM tokenizer rejects chat_template/chat_template_kwargs,
    and supports only reasoning_effort=none/high. Qwen-style models use
    chat_template_kwargs to disable thinking.
    """
    normalized = (model or "").strip().lower()
    if normalized.startswith("mistral") or "/mistral" in normalized:
        return {"reasoning_effort": "none"}
    return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


class ThinkingTolerantClient(OpenAIGenericClient):
    async def _generate_response(
        self,
        messages,
        response_model=None,
        max_tokens: int = 16384,
        model_size=None,
        **kwargs,
    ) -> dict[str, typing.Any]:
        openai_messages = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role in ("user", "system"):
                openai_messages.append({"role": m.role, "content": m.content})
        try:
            model = self.model or "qwen3.6-27b"
            response = await self.client.chat.completions.create(
                model=model,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                response_format=self._build_response_format(response_model),  # type: ignore[arg-type]
                **_request_extras_for_model(model),
            )
            result = _extract_body(response.choices[0].message)
            if not result:
                raise EmptyResponseError("LLM returned an empty response")
            return json.loads(self._strip_code_fences(result))
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f"Error in generating LLM response: {e}")
            raise
