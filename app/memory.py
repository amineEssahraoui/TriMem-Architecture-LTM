import chromadb 
import time
from app.config import settings
from rank_bm25 import BM25Okapi

class EpisodicMemory: 
    def __init__(self):
        # Initialize the ChromaDB client with persistent storage on disk
        self.client = chromadb.PersistentClient(path = settings.CHROMA_PATH)
        # Create or retrieve the collection for episodic memory (cos similarity)
        self.collection = self.client.get_or_create_collection(
            name = "episodic_memory", 
            metadata={"hnsw:space": "cosine"}
        )

    def add_interaction(self, user_id: str, user_message: str, ai_response: str, embedding: list[float]):
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
            embeddings = [embedding],
            documents = [content], 
            metadatas = [{"user_id": user_id, "timestamp": timestamp}] # Metadata helps with filtering
        )

    def retrieve_recent_context(self, user_id: str, query_embedding: list[float], query_text: str, n_results: int = settings.MEMORY_N_RESULTS) -> str:
        """
        Retrieves context using HYBRID SEARCH (Semantic Vectors + BM25 Keywords).
        """
        # SEMANTIC SEARCH (ChromaDB Vectors)
        semantic_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"user_id": user_id}
        )
        
        semantic_docs = []
        if semantic_results['documents'] and semantic_results['documents'][0]:
            semantic_docs = semantic_results['documents'][0]

        # KEYWORD SEARCH (BM25)
        # Fetch all user documents to run keyword search
        all_user_data = self.collection.get(where={"user_id": user_id})
        bm25_docs = []
        
        if all_user_data['documents']:
            all_docs = all_user_data['documents']
            # Tokenize the documents (split into words and convert to lowercase)
            tokenized_corpus = [doc.lower().split(" ") for doc in all_docs]
            
            # Initialize BM25 with the user's history
            bm25 = BM25Okapi(tokenized_corpus)
            
            # Tokenize the query and get top N matches
            tokenized_query = query_text.lower().split(" ")
            bm25_docs = bm25.get_top_n(tokenized_query, all_docs, n=n_results)

        # COMBINE & DEDUPLICATE RESULTS (Hybrid Merge)
        # We use a set to ensure we don't send the same memory twice to the LLM
        combined_docs = list(set(semantic_docs + bm25_docs))
        
        # If nothing found at all
        if not combined_docs:
            return ""

        # Return the final combined context
        return "\n\n".join(combined_docs)

# Create a global instance to be used across the application
memory_db = EpisodicMemory()