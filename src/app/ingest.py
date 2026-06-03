import requests

from app.config import DATA_SOURCE_URL

COURSES_URL = "https://datatalks.club/faq/json/courses.json"
FAQ_URL_PREFIX = "https://datatalks.club/faq"


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


def main() -> None:
    raise NotImplementedError("load docs, index them into the knowledge base")


if __name__ == "__main__":
    main()
