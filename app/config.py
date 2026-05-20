import os
from dotenv import load_dotenv

# Load environment variables if a .env file is present
load_dotenv()

class Config:
    # Language model for text generation
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3:8b")
    
    # Model for creating text embeddings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
    
    # Directory where ChromaDB will persist its data
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")

    # Number of recent context messages to retrieve (converted to integer)
    MEMORY_N_RESULTS = int(os.getenv("MEMORY_N_RESULTS", 3))

    # Simple LLM used to score episodic memories
    SCORING_MODEL = os.getenv("SCORING_MODEL", "llama3.2:latest")

# Instantiate the configuration for easy import
settings = Config()