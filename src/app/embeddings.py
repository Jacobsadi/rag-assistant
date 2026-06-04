from functools import lru_cache
from pathlib import Path

from tqdm import tqdm

from app.config import EMBEDDING_BACKEND, ONNX_MODEL_PATH

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
ENCODE_BATCH_SIZE = 50


@lru_cache(maxsize=1)
def get_embedder():
    if EMBEDDING_BACKEND == "torch":
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(EMBEDDING_MODEL)

    from app.embed.onnx_embedder import Embedder

    path = Path(ONNX_MODEL_PATH)
    if not (path / "model.onnx").exists() or not (path / "tokenizer.json").exists():
        raise FileNotFoundError(
            f"ONNX model not found at {path}. Run: uv run python -m app.embed.download"
        )
    return Embedder(path)


def document_text(doc: dict) -> str:
    return doc["question"] + " " + doc["answer"]


def vec_to_str(vector) -> str:
    return "[" + ",".join(str(float(x)) for x in vector) + "]"


def _encode_batch_onnx(embedder, texts: list[str]):
    vectors = []
    batch_range = range(0, len(texts), ENCODE_BATCH_SIZE)
    iterator = batch_range
    if len(texts) > ENCODE_BATCH_SIZE:
        iterator = tqdm(batch_range, desc="Batches")
    for i in iterator:
        batch = texts[i : i + ENCODE_BATCH_SIZE]
        vectors.extend(embedder.encode_batch(batch))
    return vectors


def embed_texts(texts: list[str]) -> list:
    embedder = get_embedder()
    if EMBEDDING_BACKEND == "torch":
        return embedder.encode(
            texts,
            batch_size=ENCODE_BATCH_SIZE,
            show_progress_bar=len(texts) > ENCODE_BATCH_SIZE,
        )
    return _encode_batch_onnx(embedder, texts)


def embed_query(query: str):
    embedder = get_embedder()
    if EMBEDDING_BACKEND == "torch":
        return embedder.encode(query)
    return embedder.encode(query)
