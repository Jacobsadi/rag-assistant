from typing import Any

from openai import OpenAI

from app.agent import SEARCH_TOOL, make_call
from app.config import LLM_MODEL, OPENAI_API_KEY

MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
    "gpt-5.4-mini": (0.75 / 1_000_000, 4.50 / 1_000_000),
}


def _client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    input_price, output_price = MODEL_PRICES.get(model, MODEL_PRICES["gpt-4o-mini"])
    return input_tokens * input_price + output_tokens * output_price


def response_usage(response, model: str) -> dict[str, Any]:
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": response.usage.total_tokens,
        "cost": _compute_cost(model, input_tokens, output_tokens),
    }


def create_response(
    messages: list,
    model: str | None = None,
    tools: list | None = None,
    temperature: float = 0.0,
):
    model_name = model or LLM_MODEL
    kwargs: dict[str, Any] = {"model": model_name, "input": messages}
    if tools is not None:
        kwargs["tools"] = tools
    if temperature != 0.0:
        kwargs["temperature"] = temperature
    return _client().responses.create(**kwargs)


def generate_without_tools(question: str, model: str | None = None) -> dict:
    messages = [{"role": "user", "content": question}]
    model_name = model or LLM_MODEL
    response = create_response(messages, model=model_name)
    return {
        "answer": response.output_text,
        "model": model_name,
        "response": response,
        **response_usage(response, model_name),
    }


def generate_with_tools(
    question: str,
    tools: list | None = None,
    model: str | None = None,
) -> dict:
    tools = tools if tools is not None else [SEARCH_TOOL]
    model_name = model or LLM_MODEL
    messages: list = [{"role": "user", "content": question}]

    response1 = create_response(messages, model=model_name, tools=tools)
    usage1 = response_usage(response1, model_name)
    messages.extend(response1.output)

    function_calls = [item for item in response1.output if item.type == "function_call"]
    if not function_calls:
        return {
            "answer": response1.output_text,
            "model": model_name,
            "response": response1,
            "turns": [usage1],
            **usage1,
        }

    for call in function_calls:
        messages.append(make_call(call))

    response2 = create_response(messages, model=model_name, tools=tools)
    usage2 = response_usage(response2, model_name)

    return {
        "answer": response2.output_text,
        "model": model_name,
        "response": response2,
        "turns": [usage1, usage2],
        "input_tokens": usage1["input_tokens"] + usage2["input_tokens"],
        "output_tokens": usage1["output_tokens"] + usage2["output_tokens"],
        "total_tokens": usage1["total_tokens"] + usage2["total_tokens"],
        "cost": usage1["cost"] + usage2["cost"],
    }


def generate(
    prompt: str,
    instructions: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict:
    model_name = model or LLM_MODEL
    message_history = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": prompt},
    ]
    response = create_response(message_history, model=model_name, temperature=temperature)
    return {
        "answer": response.output_text,
        "model": model_name,
        **response_usage(response, model_name),
    }
