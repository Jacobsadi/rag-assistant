# TODO (Module 4 lessons 02 + 03): generate a ground-truth dataset.
# For each document, ask an LLM (structured output) to produce N questions a
# user might ask. Save as (question, doc_id) pairs -> data/ground_truth.csv.
# This file powers retrieval_eval.ipynb (Hit Rate / MRR).


def generate_questions(document: dict, n: int = 5) -> list[str]:
    raise NotImplementedError("Module 4: LLM structured output -> questions")


def main() -> None:
    raise NotImplementedError("Module 4: loop over docs, save ground truth")


if __name__ == "__main__":
    main()
