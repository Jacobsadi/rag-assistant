import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from app.config import LLM_MODEL, OPENAI_API_KEY
from app.ingest import DEFAULT_COURSE, load_saved_documents
from app.llm import _compute_cost

INSTRUCTIONS_TEMPLATE = """
You emulate a student who's taking our course.
Formulate {n} questions this student might ask based on a FAQ record. The record
should contain the answer to the questions, and the questions should be complete and not too short.
If possible, use as fewer words as possible from the record.

The output should resemble how people ask questions
on the internet. Not too formal, not too short, not too long.
""".strip()

GROUND_TRUTH_PATH = Path("data/ground_truth.csv")
DEFAULT_WORKERS = 6
DEFAULT_QUESTIONS = 5
MAX_RETRIES = 3


class Questions(BaseModel):
    questions: list[str] = Field(description="Student questions that the FAQ record answers")


def _client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def _call_structured(document: dict, n: int) -> tuple[Questions, object]:
    instructions = INSTRUCTIONS_TEMPLATE.format(n=n)
    client = _client()
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.responses.parse(
                model=LLM_MODEL,
                input=[
                    {"role": "developer", "content": instructions},
                    {"role": "user", "content": json.dumps(document)},
                ],
                text_format=Questions,
            )
            if response.output_parsed is None:
                raise RuntimeError("structured output parsing failed")
            return response.output_parsed, response.usage
        except Exception as exc:
            last_error = exc
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2**attempt)

    raise last_error or RuntimeError("structured output call failed")


def generate_questions(document: dict, n: int = 5) -> list[str]:
    result, _ = _call_structured(document, n)
    return result.questions[:n]


def generate_ground_truth_records(
    document: dict,
    n: int = DEFAULT_QUESTIONS,
) -> tuple[list[dict], object]:
    result, usage = _call_structured(document, n)
    records = [
        {"question": question, "document": document["id"]}
        for question in result.questions[:n]
    ]
    return records, usage


def map_progress(pool: ThreadPoolExecutor, seq: list, fn) -> list:
    results = []
    with tqdm(total=len(seq)) as progress:
        futures = []
        for item in seq:
            future = pool.submit(fn, item)
            future.add_done_callback(lambda _: progress.update())
            futures.append(future)
        for future in futures:
            results.append(future.result())
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ground-truth questions for FAQ documents",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N documents",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--questions", type=int, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=GROUND_TRUTH_PATH)
    parser.add_argument("--course", type=str, default=DEFAULT_COURSE)
    args = parser.parse_args()

    documents = [doc for doc in load_saved_documents() if doc.get("course") == args.course]
    if args.limit is not None:
        documents = documents[: args.limit]

    if not documents:
        raise RuntimeError(f"no documents found for course={args.course!r}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = map_progress(
            pool,
            documents,
            lambda doc: generate_ground_truth_records(doc, n=args.questions),
        )

    ground_truth: list[dict] = []
    total_cost = 0.0
    for records, usage in results:
        ground_truth.extend(records)
        total_cost += _compute_cost(LLM_MODEL, usage.input_tokens, usage.output_tokens)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ground_truth).to_csv(args.output, index=False)

    print(f"Documents: {len(documents)}")
    print(f"Questions: {len(ground_truth)}")
    print(f"Cost: ${total_cost:.6f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
