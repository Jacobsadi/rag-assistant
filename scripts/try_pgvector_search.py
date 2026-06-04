from app.retrieval.vector import VectorRetriever

QUERY = "the program has already begun, can I still sign up?"


def main() -> None:
    retriever = VectorRetriever()
    try:
        hits = retriever.search(QUERY, k=5)
        for doc in hits:
            print(f"[{doc['course']}] {doc['question']}")
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
