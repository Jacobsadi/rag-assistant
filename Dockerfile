FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install uv

COPY pyproject.toml ./
RUN uv pip install --system -r pyproject.toml

COPY src/ ./src/
COPY data/ ./data/

ENV PYTHONPATH=/app/src

EXPOSE 8000 8501

CMD ["streamlit", "run", "src/app/ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
