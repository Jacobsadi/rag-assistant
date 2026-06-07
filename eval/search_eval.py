from collections.abc import Callable
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from app.ingest import DEFAULT_COURSE, load_saved_documents

GROUND_TRUTH_PATH = Path("data/ground_truth.csv")
SearchFn = Callable[[str], list[dict]]


def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> list[dict]:
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def load_course_documents(course: str = DEFAULT_COURSE) -> list[dict]:
    return [doc for doc in load_saved_documents() if doc.get("course") == course]


def make_keyword_search(documents: list[dict], k: int = 5) -> SearchFn:
    from app.retrieval.keyword import KeywordRetriever

    retriever = KeywordRetriever()
    retriever.index(documents)
    return lambda query: retriever.search(query, k=k)


def make_vector_search(k: int = 5, course: str = DEFAULT_COURSE) -> SearchFn:
    from app.retrieval.vector import VectorRetriever

    retriever = VectorRetriever(course=course)
    return lambda query: retriever.search(query, k=k)


def compute_relevance(record: dict, search_function: SearchFn) -> list[int]:
    doc_id = record["document"]
    results = search_function(record["question"])
    return [int(doc["id"] == doc_id) for doc in results]


def compute_relevance_total(
    ground_truth: list[dict],
    search_function: SearchFn,
    *,
    sample: int | None = None,
) -> list[list[int]]:
    records = ground_truth[:sample] if sample is not None else ground_truth
    relevance_total = []
    for record in tqdm(records):
        relevance_total.append(compute_relevance(record, search_function))
    return relevance_total
