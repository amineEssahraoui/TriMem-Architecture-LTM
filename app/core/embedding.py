import ollama
from app.core.config import settings

def get_embedding(text : str) -> list[float]: # Take a text and return emmbedding vector
    """
    Transforms text into a numerical vector (embedding) using nomic-embed-text.
    """
    try:
        response = ollama.embeddings(
            model=settings.EMBEDDING_MODEL,
            prompt=text
        )
        return response['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []