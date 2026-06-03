import os

from dotenv import load_dotenv

load_dotenv()

# TODO [Module 1]: load and expose your settings (LLM_MODEL, OPENAI_API_KEY, POSTGRES dict, DATA_SOURCE_URL)

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

POSTGRES = {
    "host": os.getenv("POSTGRES_HOST"),
    "port": os.getenv("POSTGRES_PORT"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "database": os.getenv("POSTGRES_DB"),
}

DATA_SOURCE_URL = os.getenv("DATA_SOURCE_URL")