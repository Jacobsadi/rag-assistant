from typing import Any

from app.agent import AGENT_INSTRUCTIONS, SEARCH_TOOL, bind_retriever, search
from app.config import LLM_MODEL

DEFAULT_MODEL = "gpt-5.4-mini"


def _require_toyaikit():
    try:
        from toyaikit.chat import IPythonChatInterface
        from toyaikit.chat.runners import DisplayingRunnerCallback, OpenAIResponsesRunner
        from toyaikit.llm import OpenAIClient
        from toyaikit.tools import Tools
    except ImportError as exc:
        raise ImportError(
            "install ToyAIKit only: uv pip install toyaikit 'openai>=1.97.0'"
        ) from exc
    return Tools, OpenAIClient, IPythonChatInterface, OpenAIResponsesRunner, DisplayingRunnerCallback


def build_tools(*, with_schema: bool = False):
    Tools, *_ = _require_toyaikit()
    agent_tools = Tools()
    if with_schema:
        agent_tools.add_tool(search, SEARCH_TOOL)
    else:
        agent_tools.add_tool(search)
    return agent_tools


def build_runner(
    instructions: str | None = None,
    model: str | None = None,
    *,
    with_schema: bool = False,
):
    (
        _Tools,
        OpenAIClient,
        IPythonChatInterface,
        OpenAIResponsesRunner,
        DisplayingRunnerCallback,
    ) = _require_toyaikit()

    agent_tools = build_tools(with_schema=with_schema)
    chat_interface = IPythonChatInterface()
    callback = DisplayingRunnerCallback(chat_interface)
    runner = OpenAIResponsesRunner(
        tools=agent_tools,
        developer_prompt=instructions or AGENT_INSTRUCTIONS,
        chat_interface=chat_interface,
        llm_client=OpenAIClient(model=model or DEFAULT_MODEL),
    )
    return runner, callback


def run_prompt(
    question: str,
    *,
    instructions: str | None = None,
    model: str | None = None,
    previous_messages: list | None = None,
    with_schema: bool = False,
):
    runner, callback = build_runner(
        instructions=instructions,
        model=model or LLM_MODEL or DEFAULT_MODEL,
        with_schema=with_schema,
    )
    return runner.loop(
        prompt=question,
        previous_messages=previous_messages,
        callback=callback,
    )


def run_chat(
    *,
    instructions: str | None = None,
    model: str | None = None,
    with_schema: bool = False,
) -> None:
    runner, _ = build_runner(
        instructions=instructions,
        model=model or LLM_MODEL or DEFAULT_MODEL,
        with_schema=with_schema,
    )
    runner.run()


def loop_result_summary(result) -> dict[str, Any]:
    tokens = result.tokens
    return {
        "answer": result.last_message,
        "cost": result.cost,
        "input_tokens": tokens.input_tokens if tokens else None,
        "output_tokens": tokens.output_tokens if tokens else None,
        "all_messages": result.all_messages,
    }
