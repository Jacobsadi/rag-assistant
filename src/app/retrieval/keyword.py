from minsearch import Index

from app.retrieval.base import Retriever

BOOST = {"question": 2.0, "section": 0.5}
DEFAULT_COURSE = "llm-zoomcamp"


class KeywordRetriever(Retriever):
    def __init__(self) -> None:
        self._index: Index | None = None

    def index(self, documents: list[dict]) -> None:
        self._index = Index(
            text_fields=["question", "section", "answer"],
            keyword_fields=["course"],
        )
        self._index.fit(documents)

    def search(
        self,
        query: str,
        k: int = 5,
        course: str = DEFAULT_COURSE,
    ) -> list[dict]:
        if self._index is None:
            raise RuntimeError("call index(documents) before search")
        return self._index.search(
            query,
            boost_dict=BOOST,
            filter_dict={"course": course},
            num_results=k,
        )
