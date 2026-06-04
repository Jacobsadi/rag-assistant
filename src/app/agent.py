import json
from typing import Any

from app.config import LLM_MODEL
from app.retrieval.keyword import KeywordRetriever

SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "search",
    "description": "Search the FAQ database for entries matching the given query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the course FAQ.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

AGENT_INSTRUCTIONS = """
You're a course teaching assistant.
You're given a question from a course student and your task is to answer it.

If you want to look up information, use the search function.
Use as many keywords from the user question as possible when making first requests.

Make multiple searches. First perform search, analyze the results
and then perform more searches.

The question has to be about the course or its logistics, offtopic questions
shouldn't be answered. If the search returns nothing, it's likely an off-topic question.
If you can't answer the question using FAQ, don't do it yourself. Only use the
facts from the FAQ database.

At the end, ask if there are other areas that the user wants to explore.
""".strip()

_bound_retriever: KeywordRetriever | None = None


def bind_retriever(retriever: KeywordRetriever) -> None:
    global _bound_retriever
    _bound_retriever = retriever


def search(query: str, k: int = 5) -> list[dict]:
    """Search the FAQ database for entries matching the given query."""
    if _bound_retriever is None:
        raise RuntimeError("call bind_retriever(retriever) before search")
    return _bound_retriever.search(query, k=k)


def make_call(call) -> dict:
    args = json.loads(call.arguments)
    if call.name != "search":
        raise ValueError(f"unknown tool: {call.name}")
    result = search(**args)
    return {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": json.dumps(result, indent=2),
    }


def _message_text(item) -> str:
    if not item.content:
        return ""
    return item.content[0].text


def process_response(
    messages: list,
    response,
    *,
    verbose: bool = False,
) -> tuple[bool, str]:
    messages.extend(response.output)
    has_function_calls = False
    last_answer = ""

    for item in response.output:
        if item.type == "function_call":
            if verbose:
                print("function_call:", item.name, item.arguments)
            messages.append(make_call(item))
            has_function_calls = True
        elif item.type == "message":
            last_answer = _message_text(item)
            if verbose:
                print("ASSISTANT:")
                print(last_answer)

    return has_function_calls, last_answer


def agent_loop(
    question: str,
    instructions: str | None = None,
    model: str | None = None,
    verbose: bool = False,
    max_iterations: int | None = None,
) -> dict:
    from app.llm import create_response, response_usage

    instructions = instructions or AGENT_INSTRUCTIONS
    model_name = model or LLM_MODEL
    tools = [SEARCH_TOOL]
    messages: list = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": question},
    ]

    last_answer = ""
    turns: list[dict[str, Any]] = []
    iteration = 1

    while True:
        if max_iterations is not None and iteration > max_iterations:
            break
        if verbose:
            print(f"iteration #{iteration}...")

        response = create_response(messages, model=model_name, tools=tools)
        turns.append(response_usage(response, model_name))

        has_function_calls, last_answer = process_response(
            messages, response, verbose=verbose
        )

        iteration += 1
        if not has_function_calls:
            break

    return {
        "answer": last_answer,
        "model": model_name,
        "messages": messages,
        "iterations": iteration - 1,
        "turns": turns,
        "input_tokens": sum(t["input_tokens"] for t in turns),
        "output_tokens": sum(t["output_tokens"] for t in turns),
        "total_tokens": sum(t["total_tokens"] for t in turns),
        "cost": sum(t["cost"] for t in turns),
    }
