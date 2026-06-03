import os

from dotenv import load_dotenv

load_dotenv()

# TODO [Module 1]: load and expose your settings (LLM_MODEL, OPENAI_API_KEY, POSTGRES dict, DATA_SOURCE_URL)

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
