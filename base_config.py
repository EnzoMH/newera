import os

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME: str = os.getenv("MODEL_NAME", "exaone")

MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "semiconductor_rag")
MONGODB_COLLECTION: str = os.getenv("MONGODB_COLLECTION", "documents")

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "data/vectordb/faiss.index")

DATA_DIR: str = os.getenv("DATA_DIR", "data/")
CRAWLED_DIR: str = os.getenv("CRAWLED_DIR", "data/crawled/")
CHUNKS_DIR: str = os.getenv("CHUNKS_DIR", "data/chunks/")