"""The agent loop, written by hand (no framework) so every step is visible.

Claude decides which tools to call; we run them and send results back, repeating until
Claude answers in plain text. Every turn is recorded in a Trace for evals and debugging.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field

import anthropic

from .prompts import SYSTEM_PROMPT
from .tools import TOOLS, run_tool

DEFAULT_MODEL = os.getenv("KISAN_MODEL", "claude-opus-5-5")
MAX_TURNS = 8  # hard stop so a confused agent can't loop (and bill) forever


@dataclass
class Trace:
    answer: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    stop_reason: str = ""


def _request_params(model: str) -> dict:
    """Model-specific options. Opus 5.5 takes effort + server-side refusal fallback; Haiku 4.5 takes neither."""
    if model.startswith("claude-opus-5"):
        return {
            "output_config": {"effort": "medium"},
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
        }
    return {}


class KisanAgent:
    def __init__(self, client: anthropic.AsyncAnthropic | None = None, model: str = DEFAULT_MODEL):
        self.client = client or anthropic.AsyncAnthropic()
        self.model = model

    async def ask(self, text: str, images: dict[str, bytes] | None = None) -> Trace:
        trace = Trace()
        start = time.perf_counter()
        content = text
        if images:
            content = f"{text}\n\n[Attached image ids: {', '.join(images)}]"
        messages: list = [{"role": "user", "content": content}]

        for _ in range(MAX_TURNS):
            response = await self.client.beta.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
                **_request_params(self.model),
            )
            trace.input_tokens += response.usage.input_tokens
            trace.output_tokens += response.usage.output_tokens
            trace.stop_reason = response.stop_reason

            if response.stop_reason != "tool_use":
                break  # end_turn, max_tokens or refusal: nothing more to execute

            messages.append({"role": "assistant", "content": response.content})
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            outputs = await asyncio.gather(*(run_tool(b.name, b.input, images) for b in tool_uses))
            for b, out in zip(tool_uses, outputs):
                trace.tool_calls.append({"name": b.name, "input": b.input, "output": out})
            # All results go back in ONE user message, which keeps parallel tool calls working.
            messages.append({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": b.id, "content": out}
                    for b, out in zip(tool_uses, outputs)
                ],
            })
        else:
            trace.stop_reason = "max_turns"

        if trace.stop_reason == "refusal":
            trace.answer = "Sorry, I can't help with that. Please contact your local KVK."
        else:
            trace.answer = "".join(b.text for b in response.content if b.type == "text").strip()
        trace.latency_s = round(time.perf_counter() - start, 2)
        return trace
