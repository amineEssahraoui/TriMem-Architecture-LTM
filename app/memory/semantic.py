import sqlite3
import os
import json
from groq import Groq
from app.core.config import settings

class SemanticMemory:
    def __init__(self):
        # Dedicated separate database file for Graph Knowledge
        self.sqlite_path = os.path.join(settings.KNOWLEDGR_GRAPHE_PATH)
        self.conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        
        # Initialize Groq using the key stored in settings
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        self._init_sqlite()

        # Define the Graph Ontology
        # THE FULL GRAPH ONTOLOGY
        self.ALL_RELATIONS = [
            # 1-TO-1 RELATIONS (Overwrites old data)
            "HAS_NAME",        # 1-to-1
            "HAS_CODE",        # 1-to-1
            "LIVES_IN",        # 1-to-1
            "STUDIES_AT",      # 1-to-1
            "HAS_ROLE",        # 1-to-1
            "USES_OS",         # 1-to-1
            
            # 1-TO-N RELATIONS (Appends new data)
            "SPEAKS_LANGUAGE", # 1-to-N
            "WORKS_AT",        # 1-to-N
            "IS_A",            # 1-to-N
            "USES_TOOL",       # 1-to-N
            "HAS_SKILL",       # 1-to-N
            "WANTS_TO_LEARN",  # 1-to-N
            "WORKS_ON",        # 1-to-N
            "HAS_GOAL",        # 1-to-N
            "INTERESTED_IN",   # 1-to-N
            "HAS_HOBBIES"      # 1-to-N 
        ]
        
        # DYNAMIC OVERWRITE BOUNDS
        self.ONE_TO_ONE_RELATIONS = [
            "HAS_NAME", "HAS_CODE", "LIVES_IN", "STUDIES_AT", "HAS_ROLE", "USES_OS"
        ]

    def _init_sqlite(self):
        """Creates the isolated triplet table for graph storage."""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS semantic_triplets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                subject TEXT,
                relation TEXT,
                object TEXT,
                source_memory_id TEXT,
                UNIQUE(user_id, relation, object) 
            )
        ''')
        self.conn.commit()

    def _get_dynamic_relations(self, user_id: str) -> list[str]:
        """Dynamic Schema Filter: Drops 1-to-1 relations from the schema if populated."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT DISTINCT relation FROM semantic_triplets WHERE user_id = ?', (user_id,))
        existing_relations = [row[0] for row in cursor.fetchall()]
        
        allowed_relations = []
        for rel in self.ALL_RELATIONS:
            # Skip 1-to-1 relations if they are already stored for this user
            if rel in self.ONE_TO_ONE_RELATIONS and rel in existing_relations:
                continue 
            allowed_relations.append(rel)
            
        return allowed_relations

    def extract_and_save_facts(self, user_id: str, text: str, source_id: str):
        """Extracts triplets via Groq and smartly handles 1-to-1 overwrites vs 1-to-N appends."""
        
        # We give the LLM the FULL list every time so it can catch updates!
        allowed = self.ALL_RELATIONS

        system_prompt = (
            "You are a strict Data Extraction AI. Extract factual triplets from the user's text.\n"
            "You MUST output valid JSON. You MUST output a single JSON object containing a 'facts' key.\n\n"
            "STRICT RULES:\n"
            "1. The 'subject' must always be exactly 'User'.\n"
            f"2. You can ONLY use the following exact relations: {json.dumps(self.ALL_RELATIONS)}\n"
            "3. If no facts fit these exact relations, you MUST output {\"facts\": []}.\n"
            "4. RESTRAINT: Do NOT force extractions from casual conversation. If the user is just chatting or making general statements, output {\"facts\": []}.\n\n"
            "EXPECTED JSON FORMAT:\n"
            "{\n"
            '  "facts": [\n'
            '    {"subject": "User", "relation": "USES_TOOL", "object": "PyTorch"}\n'
            "  ]\n"
            "}"
        )

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"}
            )
            
            parsed_json = json.loads(response.choices[0].message.content)
            facts = parsed_json.get("facts", [])

            cursor = self.conn.cursor()
            for fact in facts:
                relation = fact.get('relation')
                obj = fact.get('object')
                subject = fact.get('subject')

                if relation not in allowed:
                    print(f"[Semantic Graph] Blocked disallowed relation: {relation}")
                    continue
                
                try:
                    # 1-TO-1 OVERWRITE LOGIC
                    if relation in self.ONE_TO_ONE_RELATIONS:
                        # Delete whatever the old object was for this specific relation
                        cursor.execute('''
                            DELETE FROM semantic_triplets 
                            WHERE user_id = ? AND relation = ?
                        ''', (user_id, relation))
                        
                        # Insert the new object
                        cursor.execute('''
                            INSERT INTO semantic_triplets (user_id, subject, relation, object, source_memory_id)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (user_id, subject, relation, obj, source_id))
                        print(f"[Semantic Graph] UPDATED 1-to-1: {subject} -> {relation} -> {obj}")

                    # 1-TO-N APPEND LOGIC
                    else:
                        cursor.execute('''
                            INSERT INTO semantic_triplets (user_id, subject, relation, object, source_memory_id)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (user_id, subject, relation, obj, source_id))
                        print(f"[Semantic Graph] ADDED 1-to-N: {subject} -> {relation} -> {obj}")

                except sqlite3.IntegrityError:
                    # This safely catches if the 1-to-N fact already exists exactly as written
                    print(f"[Semantic Graph] SKIPPED (Duplicate Fact): {obj}")
                    
            self.conn.commit()

        except Exception as e:
            print(f"[Semantic Error] Fact extraction pipeline failed: {e}")

# Instantiate the singleton
semantic_memory = SemanticMemory()