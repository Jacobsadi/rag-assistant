import json
from pathlib import Path

import requests

from app.config import DATA_SOURCE_URL

COURSES_URL = DATA_SOURCE_URL or "https://datatalks.club/faq/json/courses.json"
FAQ_URL_PREFIX = "https://datatalks.club/faq"
PROCESSED_DOCUMENTS_PATH = Path("data/processed/documents.json")
DEFAULT_COURSE = "llm-zoomcamp"


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
    documents = load_documents()
    save_documents(documents)
    course_count = sum(1 for doc in documents if doc.get("course") == DEFAULT_COURSE)
    print(f"Loaded {len(documents)} documents")
    print(f"{DEFAULT_COURSE}: {course_count} documents")
    print(f"Saved to {PROCESSED_DOCUMENTS_PATH}")


if __name__ == "__main__":
    main()
