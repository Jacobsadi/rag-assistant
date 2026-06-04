import argparse
import json
from pathlib import Path

import requests

from app.config import DATA_SOURCE_URL

COURSES_URL = DATA_SOURCE_URL or "https://datatalks.club/faq/json/courses.json"
FAQ_URL_PREFIX = "https://datatalks.club/faq"
PROCESSED_DOCUMENTS_PATH = Path("data/processed/documents.json")
DEFAULT_COURSE = "llm-zoomcamp"


def index_pgvector(documents: list[dict], *, reset: bool = True) -> None:
    from app.retrieval.vector import VectorRetriever

    retriever = VectorRetriever()
    try:
        retriever.index(documents, reset=reset)
    finally:
        retriever.close()


def load_documents() -> list[dict]:
    response = requests.get(COURSES_URL, timeout=30)
    response.raise_for_status()
    courses_raw = response.json()

    documents: list[dict] = []
    for course in courses_raw:
        course_url = f"{FAQ_URL_PREFIX}{course['path']}"
        course_response = requests.get(course_url, timeout=30)
        course_response.raise_for_status()
        documents.extend(course_response.json())

    return documents


def save_documents(
    documents: list[dict],
    path: Path = PROCESSED_DOCUMENTS_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(documents, indent=2), encoding="utf-8")


def load_saved_documents(path: Path = PROCESSED_DOCUMENTS_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"run ingestion first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FAQ data and index into PGVector")
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="skip download; use data/processed/documents.json",
    )
    parser.add_argument(
        "--skip-pgvector",
        action="store_true",
        help="only download/save JSON, do not write embeddings to Postgres",
    )
    parser.add_argument(
        "--pgvector-only",
        action="store_true",
        help="only index cached JSON into Postgres (no download)",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="keep existing documents table rows (append mode; schema must exist)",
    )
    args = parser.parse_args()

    if args.pgvector_only or args.from_cache:
        documents = load_saved_documents()
    else:
        documents = load_documents()
        save_documents(documents)

    course_count = sum(1 for doc in documents if doc.get("course") == DEFAULT_COURSE)
    print(f"Documents: {len(documents)} ({DEFAULT_COURSE}: {course_count})")
    if not args.pgvector_only:
        print(f"Saved to {PROCESSED_DOCUMENTS_PATH}")

    if not args.skip_pgvector:
        print("Indexing embeddings into PGVector...")
        index_pgvector(documents, reset=not args.no_reset)
        print("PGVector index ready")


if __name__ == "__main__":
    main()
