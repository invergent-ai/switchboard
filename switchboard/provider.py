"""Minimal async chat client for any OpenAI-compatible endpoint (OpenRouter by default)."""

from __future__ import annotations

import asyncio
import os

from openai import AsyncOpenAI

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class Provider:
    """Bounded-concurrency async chat completion. The API key is read from an env var only."""

    def __init__(
        self,
        model: str,
        concurrency: int = 10,
        base_url: str = OPENROUTER_BASE,
        api_key_env: str = "OPENROUTER_API_KEY",
    ) -> None:
        self.model = model
        self._sem = asyncio.Semaphore(concurrency)
        self._client = AsyncOpenAI(api_key=os.environ[api_key_env], base_url=base_url)

    async def _one(self, messages: list[dict], max_tokens: int) -> str:
        async with self._sem:
            for attempt in range(4):
                try:
                    r = await self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0,
                    )
                    return r.choices[0].message.content or ""
                except Exception:  # noqa: BLE001 — retry with backoff, then give up gracefully
                    await asyncio.sleep(min(2**attempt, 20))
            return ""

    async def complete_many(self, batch: list[list[dict]], max_tokens: int = 400) -> list[str]:
        return await asyncio.gather(*(self._one(m, max_tokens) for m in batch))
