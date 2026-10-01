"""Chat with the agent in the terminal:  kisan "Sehore mein gehun ka bhav kya hai?" """

from __future__ import annotations

import asyncio
import sys

from .agent import KisanAgent


def main() -> None:
    question = " ".join(sys.argv[1:]) or input("Aapka sawal: ")
    trace = asyncio.run(KisanAgent().ask(question))
    for call in trace.tool_calls:
        print(f"  [tool] {call['name']}({call['input']})", file=sys.stderr)
    print(trace.answer)
    print(f"  [{trace.input_tokens} in / {trace.output_tokens} out tokens, {trace.latency_s}s]", file=sys.stderr)


if __name__ == "__main__":
    main()
