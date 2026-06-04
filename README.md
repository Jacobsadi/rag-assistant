# RAG Assistant — Retrieval-Augmented Q&A for `<DOMAIN>`

> An end-to-end Retrieval-Augmented Generation (RAG) application: automated ingestion,
> hybrid retrieval with reranking, offline + online evaluation, a FastAPI/Streamlit
> interface, and live monitoring with Grafana.

<!-- Replace placeholders marked with <...>. Add a demo gif/screenshot below. -->

![demo](docs/images/demo.gif)

## Problem

`<Describe the problem this assistant solves and who it is for. 2-3 sentences.
Make it clear what question a user asks and what grounded answer they get back.>`

## Architecture

![architecture](docs/architecture.png)

```
Data source ──(dlt/script)──▶ Knowledge base (PGVector)
                                     ▲
User ──▶ UI/API ──▶ Hybrid retrieval + rerank ──┘
                       │
                       ▼
                 Prompt builder ──▶ LLM ──▶ Answer ──▶ User
                       │
                       ▼
            Postgres (conversations + feedback) ──▶ Grafana
```

See [docs/architecture.md](docs/architecture.md) for details.

## Dataset

- Source: `<url / how obtained>`
- Size / license: `<...>`
- Reproduce: `make download-onnx-model` then `make ingest` (ONNX `Xenova/all-MiniLM-L6-v2` → PGVector)

## Quickstart

```bash
cp .env.example .env          # add your OPENAI_API_KEY
docker compose up -d          # app, postgres+pgvector, grafana
make ingest                   # build the knowledge base
# UI:      http://localhost:8501
# API:     http://localhost:8000/docs
# Grafana: http://localhost:3000
```

Local dev without Docker:

```bash
uv sync --extra dev
make run        # streamlit UI
make api        # fastapi
```

## Retrieval & Evaluation

Retrieval approaches compared in [`eval/retrieval_eval.ipynb`](eval/retrieval_eval.ipynb):

| Approach | Hit Rate | MRR |
|----------|----------|-----|
| Keyword  | `<...>`  | `<...>` |
| Vector   | `<...>`  | `<...>` |
| Hybrid + rerank | `<...>` | `<...>` |

Answer quality (LLM-as-a-judge, multiple prompts/models) in
[`eval/rag_eval.ipynb`](eval/rag_eval.ipynb). Full write-up:
[docs/evaluation.md](docs/evaluation.md).

## Monitoring

User feedback (thumbs up/down) and system metrics are stored in Postgres and
visualized in Grafana (5+ panels: volume, latency, cost, feedback, relevance).

![dashboard](docs/images/dashboard.png)

## Tech Stack

LLM: OpenAI · Retrieval: minsearch / PGVector hybrid + rerank · Ingestion: dlt ·
Interface: FastAPI + Streamlit · Storage: PostgreSQL · Monitoring: Grafana ·
Containerization: Docker Compose.

## Project Structure

```
src/app/      application code (ingest, rag, retrieval, llm, db, api, ui)
eval/         ground truth + retrieval/RAG evaluation notebooks
grafana/      provisioned datasource + dashboards
tests/        unit tests
docs/         architecture + evaluation write-ups
```

## Roadmap

- [ ] Query rewriting
- [ ] Cloud deployment
- [ ] Streaming responses

## License

MIT — see [LICENSE](LICENSE).
