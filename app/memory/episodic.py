import chromadb 
import time
import math
import sqlite3
import os
from app.core.config import settings

class EpisodicMemory: 
    def __init__(self):
        # Initialize ChromaDB for Semantic Search (Vectors)
        self.client = chromadb.PersistentClient(path = settings.CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name = "episodic_memory", 
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize SQLite FTS5 for Keyword Search (Text)
        os.makedirs(settings.CHROMA_PATH, exist_ok=True)
        self.sqlite_path = os.path.join(settings.CHROMA_PATH, "keywords.db")
        
        self.conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        self._init_sqlite()

    def _init_sqlite(self):
        """Creates the Full-Text Search table if it doesn't exist."""
        cursor = self.conn.cursor()
        # FTS5 allows blazing fast keyword search on the 'content' column.
        # We store metadata as UNINDEXED so it doesn't interfere with the text search.
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                user_id UNINDEXED,
                timestamp UNINDEXED,
                importance_score UNINDEXED,
                content
            )
        ''')
        self.conn.commit()

    def add_interaction(self, user_id: str, user_message: str, ai_response: str, embedding: list[float], importance_score: int = 5):
        """Stores a single user-AI interaction in both databases."""
        timestamp = str(time.time())
        memory_id = f"{user_id}_{timestamp}"
        content = f"User: {user_message}\nAI: {ai_response}"
        
        # Save to ChromaDB (Semantic)
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

        # Save to SQLite FTS5 (Keyword)
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO memories_fts (user_id, timestamp, importance_score, content)
            VALUES (?, ?, ?, ?)
        ''', (user_id, timestamp, importance_score, content))
        self.conn.commit()

    def _calculate_final_score(self, importance: int, timestamp: float) -> float:
        """Calculates the final memory score using Importance and Time Decay."""
        age_in_seconds = time.time() - timestamp
        age_in_days = age_in_seconds / 86400.0
        decay_rate = 0.02
        decay_factor = math.exp(-decay_rate * age_in_days)
        return importance * decay_factor

    def retrieve_recent_context(self, user_id: str, query_embedding: list[float], query_text: str, n_results: int = settings.MEMORY_N_RESULTS) -> str:
        """Retrieves and ranks context using Hybrid Search (Chroma + SQLite)."""
        
        # Dictionary to store and deduplicate results: { "content": final_score }
        unique_memories = {} 

        # SEMANTIC SEARCH (ChromaDB)
        semantic_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"user_id": user_id},
            include=["documents", "metadatas"]
        )
        
        if semantic_results['documents'] and semantic_results['documents'][0]:
            docs = semantic_results['documents'][0]
            metas = semantic_results['metadatas'][0]
            for doc, meta in zip(docs, metas):
                score = self._calculate_final_score(meta["importance_score"], float(meta["timestamp"]))
                unique_memories[doc] = score

        # KEYWORD SEARCH (SQLite FTS5)
        # Clean the query string to avoid SQL syntax errors (keep only alphanumeric words)
        safe_words = [word for word in query_text.replace('"', '').split() if word.isalnum()]
        
        if safe_words:
            # Create an FTS5 OR match query (e.g., "word1" OR "word2")
            match_query = " OR ".join(f'"{w}"' for w in safe_words)
            
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT timestamp, importance_score, content 
                FROM memories_fts 
                WHERE user_id = ? AND memories_fts MATCH ?
                LIMIT ?
            ''', (user_id, match_query, n_results))
            
            keyword_results = cursor.fetchall()
            
            for timestamp, importance, content in keyword_results:
                # If ChromaDB already found this exact memory, we skip it (Deduplication)
                if content not in unique_memories: 
                    score = self._calculate_final_score(importance, float(timestamp))
                    unique_memories[content] = score

        # MERGE & RANK
        if not unique_memories:
            return ""

        # Sort the dictionary by the highest score
        ranked_docs = sorted(unique_memories.keys(), key=lambda d: unique_memories[d], reverse=True)

        return "\n\n".join(ranked_docs[:n_results])

episodic_memory = EpisodicMemory()