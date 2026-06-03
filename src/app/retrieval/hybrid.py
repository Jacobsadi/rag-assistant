from app.retrieval.base import Retriever

# TODO [Module 6]: combine keyword + vector results and rerank (e.g. Reciprocal Rank Fusion)


class HybridRetriever(Retriever):
    def index(self, documents: list[dict]) -> None:
        raise NotImplementedError("Module 6: index both keyword and vector retrievers")

    def search(self, query: str, k: int = 5) -> list[dict]:
        raise NotImplementedError("Module 6: fuse + rerank keyword and vector results")
