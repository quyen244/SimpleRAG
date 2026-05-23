from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

OLLAMA_MODEL = "qwen3.5:4b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

RETRIEVER_K = 5
RETRIEVER_FETCH_K = 10

MAX_HISTORY_TURNS = 5
