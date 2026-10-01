"""Bridges kisan-agent to agent-evals:  agent-evals run evals/cases.yaml --target evals.adapter:run"""

from agent_evals import AgentResult

from kisan_agent.agent import KisanAgent

_agent = KisanAgent()


async def run(question: str) -> AgentResult:
    t = await _agent.ask(question)
    return AgentResult(
        answer=t.answer,
        tool_calls=[c["name"] for c in t.tool_calls],
        input_tokens=t.input_tokens,
        output_tokens=t.output_tokens,
        latency_s=t.latency_s,
    )
