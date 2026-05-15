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

# Instantiate the configuration for easy import
settings = Config()