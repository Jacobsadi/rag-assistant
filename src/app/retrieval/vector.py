from app.embeddings import document_text, embed_query, embed_texts, vec_to_str
from app.ingest import DEFAULT_COURSE
from app.retrieval.base import Retriever
from app.retrieval.pgvector_store import (
    connect,
    create_hnsw_index,
    init_schema,
    insert_documents,
    search_documents,
)


class VectorRetriever(Retriever):
    def __init__(self, course: str = DEFAULT_COURSE) -> None:
        self.course = course
        self._conn = None

    def _connection(self):
        if self._conn is None or self._conn.closed:
            self._conn = connect()
        return self._conn

    def index(self, documents: list[dict], *, reset: bool = True) -> None:
        conn = self._connection()
        init_schema(conn, reset=reset)
        texts = [document_text(doc) for doc in documents]
        vectors = embed_texts(texts)
        insert_documents(conn, documents, vectors)
        create_hnsw_index(conn)

    def search(
        self,
        query: str,
        k: int = 5,
        course: str | None = None,
    ) -> list[dict]:
        course = course or self.course
        query_str = vec_to_str(embed_query(query))
        return search_documents(
            self._connection(),
            query_str,
            course=course,
            k=k,
        )

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
            self._conn = None
