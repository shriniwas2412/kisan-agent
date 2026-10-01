"""Agent-loop tests with a fake Anthropic client: no API key, no network, no cost."""

import json
from types import SimpleNamespace as NS

from kisan_agent import tools, vision
from kisan_agent.agent import KisanAgent


def text(t):
    return NS(type="text", text=t)


def tool_use(id, name, input):
    return NS(type="tool_use", id=id, name=name, input=input)


def reply(stop_reason, *content):
    return NS(stop_reason=stop_reason, content=list(content), usage=NS(input_tokens=100, output_tokens=20))


class FakeClient:
    """Replays scripted responses and records every request it receives."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = NS(messages=NS(create=self._create))

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


async def test_calls_tool_then_answers():
    client = FakeClient(
        reply("tool_use", tool_use("t1", "crop_calendar", {"crop": "chana"})),
        reply("end_turn", text("Chana October-November mein boyein.")),
    )
    trace = await KisanAgent(client, model="claude-haiku-4-5").ask("Chana kab boyein?")

    assert trace.answer == "Chana October-November mein boyein."
    assert [c["name"] for c in trace.tool_calls] == ["crop_calendar"]
    assert json.loads(trace.tool_calls[0]["output"])["crop"] == "chickpea"
    assert trace.input_tokens == 200
    # second request carries the tool result back, matched by id
    last_msg = client.requests[1]["messages"][-1]
    assert last_msg["content"][0]["tool_use_id"] == "t1"


async def test_parallel_tool_results_in_one_message():
    client = FakeClient(
        reply("tool_use", tool_use("a", "crop_calendar", {"crop": "wheat"}), tool_use("b", "what_to_sow", {"month": 10})),
        reply("end_turn", text("ok")),
    )
    await KisanAgent(client, model="claude-haiku-4-5").ask("q")
    results = client.requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]


async def test_haiku_gets_no_opus_only_params():
    client = FakeClient(reply("end_turn", text("hi")))
    await KisanAgent(client, model="claude-haiku-4-5").ask("hello")
    assert "output_config" not in client.requests[0] and "fallbacks" not in client.requests[0]


async def test_opus_gets_effort_and_fallback():
    client = FakeClient(reply("end_turn", text("hi")))
    await KisanAgent(client, model="claude-opus-5-5").ask("hello")
    assert client.requests[0]["output_config"] == {"effort": "medium"}
    assert client.requests[0]["fallbacks"] == "default"


async def test_refusal_gives_safe_message():
    client = FakeClient(reply("refusal"))
    trace = await KisanAgent(client, model="claude-haiku-4-5").ask("x")
    assert "KVK" in trace.answer


async def test_stops_after_max_turns():
    loop = [reply("tool_use", tool_use(f"t{i}", "what_to_sow", {})) for i in range(10)]
    trace = await KisanAgent(FakeClient(*loop), model="claude-haiku-4-5").ask("x")
    assert trace.stop_reason == "max_turns"


async def test_unknown_tool_and_missing_image_return_errors():
    assert "Unknown tool" in await tools.run_tool("nope", {})
    assert "No image" in await tools.run_tool("diagnose_leaf", {"image_id": "x"}, {})


async def test_leaf_without_model_is_honest(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "MODEL_DIR", tmp_path)  # empty dir, regardless of local models/
    out = json.loads(await tools.run_tool("diagnose_leaf", {"image_id": "img1"}, {"img1": b"fake"}))
    assert out["available"] is False
