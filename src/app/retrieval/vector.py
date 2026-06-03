from app.retrieval.base import Retriever

# TODO [Module 2]: embed documents and search by cosine similarity
# TODO [Module 2]: start with numpy/minsearch, then move to PGVector for a persistent index


class VectorRetriever(Retriever):
    def index(self, documents: list[dict]) -> None:
        raise NotImplementedError("Module 2: embed and store document vectors")

    def search(self, query: str, k: int = 5) -> list[dict]:
        raise NotImplementedError("Module 2: embed query and find nearest vectors")
