# TODO [Module 1]: assemble the RAG flow: retrieve -> build_prompt -> generate
# TODO [Module 1]: reuse this from api.py and ui.py


def build_prompt(question: str, documents: list[dict]) -> str:
    raise NotImplementedError("Module 1: combine search results into a prompt")


def rag(question: str, retriever, k: int = 5) -> dict:
    raise NotImplementedError("Module 1: retrieve, build prompt, call the LLM")
