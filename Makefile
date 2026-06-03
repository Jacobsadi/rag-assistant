.PHONY: install up down ingest eval run api lint test fmt

install:
	uv sync --extra dev

up:
	docker compose up -d

down:
	docker compose down

ingest:
	uv run python -m app.ingest

eval:
	uv run jupyter nbconvert --to notebook --execute eval/retrieval_eval.ipynb --output retrieval_eval.ipynb
	uv run jupyter nbconvert --to notebook --execute eval/rag_eval.ipynb --output rag_eval.ipynb

run:
	uv run streamlit run src/app/ui.py

api:
	uv run uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

test:
	uv run pytest -q
