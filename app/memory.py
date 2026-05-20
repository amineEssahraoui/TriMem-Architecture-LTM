import chromadb 
import time
import math
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

    def add_interaction(self, user_id: str, user_message: str, ai_response: str, embedding: list[float], importance_score: int = 5):
        """Stores a single user-AI interaction with its importance score."""
        timestamp = str(time.time())
        memory_id = f"{user_id}_{timestamp}"
        content = f"User: {user_message}\nAI: {ai_response}"
        
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                "user_id": user_id, 
                "timestamp": timestamp,
                "importance_score": importance_score
            }]
        )

    def _calculate_final_score(self, metadata: dict) -> float:
        """Calculates the final memory score using Importance and Time Decay."""
        if not metadata:
            return 0.0

        importance = metadata.get("importance_score", 5)
        timestamp = float(metadata.get("timestamp", time.time()))
        
        # Calculate age of the memory in days
        age_in_seconds = time.time() - timestamp
        age_in_days = age_in_seconds / 86400.0
        
        # Exponential decay formula
        # The older the memory, the smaller the decay_factor becomes.
        decay_rate = 0.02
        decay_factor = math.exp(-decay_rate * age_in_days)
        
        return importance * decay_factor

    def retrieve_recent_context(self, user_id: str, query_embedding: list[float], query_text: str, n_results: int = settings.MEMORY_N_RESULTS) -> str:
        """Retrieves and ranks context using Hybrid Search and Time-Weighted Scoring."""
        
        # Semantic Search
        semantic_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"user_id": user_id},
            include=["documents", "metadatas"]
        )
        
        candidate_docs = []
        candidate_metadatas = []
        
        if semantic_results['documents'] and semantic_results['documents'][0]:
            candidate_docs.extend(semantic_results['documents'][0])
            candidate_metadatas.extend(semantic_results['metadatas'][0])

        # Keyword Search (BM25)
        all_user_data = self.collection.get(
            where={"user_id": user_id},
            include=["documents", "metadatas"]
        )
        
        if all_user_data['documents']:
            all_docs = all_user_data['documents']
            all_metas = all_user_data['metadatas']
            
            tokenized_corpus = [doc.lower().split() for doc in all_docs]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = query_text.lower().split()
            
            doc_scores = bm25.get_scores(tokenized_query)
            
            # Get top N indices from BM25
            top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:n_results]
            
            for idx in top_n_indices:
                if doc_scores[idx] > 0: # Only add if it actually matches keywords
                    candidate_docs.append(all_docs[idx])
                    candidate_metadatas.append(all_metas[idx])

        # Deduplicate and Apply Time/Importance Scoring
        unique_memories = {}
        for doc, meta in zip(candidate_docs, candidate_metadatas):
            if doc not in unique_memories:
                final_score = self._calculate_final_score(meta)
                unique_memories[doc] = final_score

        # Sort memories by the highest final score
        ranked_docs = sorted(unique_memories.keys(), key=lambda d: unique_memories[d], reverse=True)

        if not ranked_docs:
            return ""

        # Return only the top n_results after re-ranking
        return "\n\n".join(ranked_docs[:n_results])

memory_db = EpisodicMemory()