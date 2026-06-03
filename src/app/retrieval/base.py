from abc import ABC, abstractmethod

# Interface for all retrievers. Implement the concrete classes yourself.


class Retriever(ABC):
    @abstractmethod
    def index(self, documents: list[dict]) -> None:
        ...

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[dict]:
        ...
