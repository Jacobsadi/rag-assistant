# TODO [Module 1 + Module 3]: automate ingestion
# TODO [Module 1 + Module 3]: download the dataset, clean/chunk it, build the knowledge base (PGVector)
# TODO [Module 1 + Module 3]: run via `make ingest`


def load_documents() -> list[dict]:
    raise NotImplementedError("download + parse your dataset into dicts")


def main() -> None:
    raise NotImplementedError("load docs, index them into the knowledge base")


if __name__ == "__main__":
    main()
