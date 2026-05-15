import chromadb 
import time
from app.config import settings

class EpisodicMemory: 
    def __init__(self):
        # Initialize the ChromaDB client with persistent storage on disk
        self.client = chromadb.PersistentClient(path = settings.CHROMA_PATH)
        # Create or retrieve the collection for episodic memory (cos similarity)
        self.collection = self.client.get_or_create_collection(
            name = "episodic_memory", 
            metadata={"hnsw:space": "cosine"}
        )

    def add_interaction(self, user_id: str, user_message: str, ai_response: str, emmbeding: list[float]): 
        """
        Stores a single user-AI interaction in the vector database.
        """
        timestamp = str(time.time())
        # Create a unique ID for this specific memory trace : user_id + time
        memory_id = f"{user_id}_{timestamp}"
        # Combine the user message and AI response to save the full context
        content = f"User: {user_message}\n AI: {ai_response}"
        self.collection.add(
            ids = [memory_id], 
            emmbeding = [emmbeding],
            documents = [content], 
            metadatas = [{"user_id": user_id, "timestamp": timestamp}] # Metadata helps with filtering
        )

    def retrieve_recent_context(self, user_id: str, query_embedding: list[float], n_results: int = settings.MEMORY_N_RESULTS) -> str:
        """
        Retrieves the most semantically relevant past interactions for a specific user.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"user_id": user_id} # Crucial: Only search memories belonging to this user
        )
        
        # If no memories are found, return an empty string
        if not results['documents'] or not results['documents'][0]:
            return ""
            
        # Combine the retrieved memory texts into a single string
        context = "\n\n".join(results['documents'][0])
        return context

# Create a global instance to be used across the application
memory_db = EpisodicMemory()