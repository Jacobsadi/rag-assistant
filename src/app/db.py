# TODO [Module 5]: init_db() - create `conversations` and `feedback` tables
# TODO [Module 5]: save_conversation(record) - persist question/answer/model/tokens/cost/relevance
# TODO [Module 5]: save_feedback(conversation_id, feedback) - store thumbs up/down (+1 / -1)
# TODO [Module 5]: recent_conversations(limit) - fetch rows for the dashboard


def init_db() -> None:
    raise NotImplementedError("Module 5: create the postgres tables")


def save_conversation(record: dict) -> str:
    raise NotImplementedError("Module 5: insert a conversation row")


def save_feedback(conversation_id: str, feedback: int) -> None:
    raise NotImplementedError("Module 5: insert a feedback row")


def recent_conversations(limit: int = 20) -> list[dict]:
    raise NotImplementedError("Module 5: query recent conversations")
