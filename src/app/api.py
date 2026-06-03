from fastapi import FastAPI

# TODO [Module 7]: expose the RAG flow over HTTP
# TODO [Module 7]: POST /ask -> {question} returns {answer, conversation_id}
# TODO [Module 7]: POST /feedback -> {conversation_id, feedback}; save both to the db

app = FastAPI(title="RAG Assistant")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# @app.post("/ask")
# def ask(...): ...

# @app.post("/feedback")
# def feedback(...): ...
