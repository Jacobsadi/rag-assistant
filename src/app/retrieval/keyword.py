from app.retrieval.base import Retriever

# TODO [Module 1]: keyword search with minsearch


class KeywordRetriever(Retriever):
    def index(self, documents: list[dict]) -> None:
        raise NotImplementedError("Module 1: build a minsearch index")

    def search(self, query: str, k: int = 5) -> list[dict]:
        raise NotImplementedError("Module 1: keyword search")
