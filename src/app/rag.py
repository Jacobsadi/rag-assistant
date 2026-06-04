from app.llm import generate

INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

USER_PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
"""


def build_context(documents: list[dict]) -> str:
    lines = []
    for doc in documents:
        lines.append(doc["section"])
        lines.append("Q: " + doc["question"])
        lines.append("A: " + doc["answer"])
        lines.append("")
    return "\n".join(lines).strip()


def build_prompt(question: str, documents: list[dict]) -> str:
    context = build_context(documents)
    prompt = USER_PROMPT_TEMPLATE.format(question=question, context=context)
    return prompt.strip()


def rag(question: str, retriever, k: int = 5, model: str | None = None) -> dict:
    documents = retriever.search(question, k=k)
    prompt = build_prompt(question, documents)
    result = generate(prompt, INSTRUCTIONS, model=model)
    return {
        **result,
        "question": question,
        "documents": documents,
    }
