import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    PROJECT_ROOT = Path(__file__).parent
    DOCUMENT_PATH = PROJECT_ROOT / "Amazon10k2022.pdf"
    VECTOR_DB_PATH = PROJECT_ROOT / "chroma_db"
    DATA_PATH = PROJECT_ROOT / "data"
    
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    COLLECTION_NAME = "amazon_10k_2022"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    VECTOR_DB_PATH.mkdir(exist_ok=True)
    DATA_PATH.mkdir(exist_ok=True)

config = Config()

if config.OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
else:
    print("OPENAI_API_KEY not found in .env file")