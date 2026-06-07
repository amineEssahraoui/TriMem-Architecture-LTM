import os
import sqlite3
import chromadb
import json
from app.core.config import settings

class ProceduralMemory:
    def __init__(self):
        # Initialize ChromaDB for Universal SOPs (Vector Database)
        self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        self.sop_collection = self.chroma_client.get_or_create_collection(
            name="procedural_sops",
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize SQLite for User-Specific Overlays (Relational Database)
        os.makedirs(settings.CHROMA_PATH, exist_ok=True)
        self.sqlite_path = os.path.join(settings.CHROMA_PATH, "procedural_rules.db")
        self.conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        self._init_sqlite()

    def _init_sqlite(self):
        """Creates the table for user-specific procedural overlays."""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                target_sop_id TEXT,  -- Can be a specific SOP ID (e.g., 'SOP_PAPER_ANALYSIS') or 'GLOBAL'
                instruction TEXT,
                UNIQUE(user_id, target_sop_id, instruction)
            )
        ''')
        self.conn.commit()

    def add_universal_sop(self, sop_id: str, trigger_description: str, steps: list[str], embedding: list[float]):
        """
        Saves a universal Standard Operating Procedure (SOP) to ChromaDB.
        - trigger_description: Used for vector matching (e.g., "Analyze a research paper").
        - steps: The actual instructions the LLM must follow.
        """
        self.sop_collection.add(
            ids=[sop_id],
            embeddings=[embedding],
            documents=[trigger_description], # We search against this text
            metadatas=[{"steps": json.dumps(steps)}] # Store the steps as JSON metadata
        )
        print(f"[Procedural] Added Universal SOP: {sop_id}")

    def add_user_rule(self, user_id: str, target_sop_id: str, instruction: str):
        """
        Saves a user-specific rule to override or augment a SOP.
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO user_rules (user_id, target_sop_id, instruction)
                VALUES (?, ?, ?)
            ''', (user_id, target_sop_id, instruction))
            self.conn.commit()
            print(f"[Procedural] Added Rule for {user_id} on {target_sop_id}")
        except sqlite3.IntegrityError:
            # Rule already exists exactly as written for this user/sop, skip safely
            print(f"[Procedural] Skipped (Duplicate Rule) for {user_id}")
    
    
# Instantiate the singleton
procedural_memory = ProceduralMemory()