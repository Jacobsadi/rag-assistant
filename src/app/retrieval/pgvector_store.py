import psycopg2
from psycopg2.extensions import connection as PgConnection

from app.config import POSTGRES
from app.embeddings import EMBEDDING_DIM, vec_to_str

DOCUMENTS_TABLE = "documents"


def connect() -> PgConnection:
    missing = [k for k, v in POSTGRES.items() if not v]
    if missing:
        raise RuntimeError(f"set postgres env vars: {', '.join(missing)}")
    return psycopg2.connect(
        host=POSTGRES["host"],
        port=POSTGRES["port"],
        user=POSTGRES["user"],
        password=POSTGRES["password"],
        dbname=POSTGRES["database"],
    )


def init_schema(conn: PgConnection, *, reset: bool = True) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        if reset:
            cur.execute(f"DROP TABLE IF EXISTS {DOCUMENTS_TABLE}")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {DOCUMENTS_TABLE} (
                id SERIAL PRIMARY KEY,
                course TEXT,
                section TEXT,
                question TEXT,
                answer TEXT,
                embedding vector({EMBEDDING_DIM})
            )
            """
        )
    conn.commit()


def insert_documents(
    conn: PgConnection,
    documents: list[dict],
    vectors: list,
) -> None:
    rows = [
        (
            doc["course"],
            doc["section"],
            doc["question"],
            doc["answer"],
            vec_to_str(vec),
        )
        for doc, vec in zip(documents, vectors, strict=True)
    ]
    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO {DOCUMENTS_TABLE}
                (course, section, question, answer, embedding)
            VALUES (%s, %s, %s, %s, %s::vector)
            """,
            rows,
        )
    conn.commit()


def create_hnsw_index(conn: PgConnection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
            ON {DOCUMENTS_TABLE}
            USING hnsw (embedding vector_cosine_ops)
            """
        )
    conn.commit()


def search_documents(
    conn: PgConnection,
    query_vector_str: str,
    *,
    course: str,
    k: int,
) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT course, section, question, answer
            FROM {DOCUMENTS_TABLE}
            WHERE course = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (course, query_vector_str, k),
        )
        rows = cur.fetchall()
    return [
        {
            "course": row[0],
            "section": row[1],
            "question": row[2],
            "answer": row[3],
        }
        for row in rows
    ]
