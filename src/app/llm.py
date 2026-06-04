from openai import OpenAI

from app.config import LLM_MODEL, OPENAI_API_KEY

MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
    "gpt-5.4-mini": (0.75 / 1_000_000, 4.50 / 1_000_000),
}


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    input_price, output_price = MODEL_PRICES.get(model, MODEL_PRICES["gpt-4o-mini"])
    return input_tokens * input_price + output_tokens * output_price


def generate(
    prompt: str,
    instructions: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict:
    client = OpenAI(api_key=OPENAI_API_KEY)
    model_name = model or LLM_MODEL
    message_history = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": prompt},
    ]
    kwargs: dict = {"model": model_name, "input": message_history}
    if temperature != 0.0:
        kwargs["temperature"] = temperature
    response = client.responses.create(**kwargs)
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    return {
        "answer": response.output_text,
        "model": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": response.usage.total_tokens,
        "cost": _compute_cost(model_name, input_tokens, output_tokens),
    }
