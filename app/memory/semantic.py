import sqlite3
import os
import json
from groq import Groq
from app.core.config import settings
import numpy as np

class SemanticMemory:
    def __init__(self):
        # Dedicated separate database file for Graph Knowledge
        self.sqlite_path = os.path.join(settings.KNOWLEDGR_GRAPHE_PATH)
        self.conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        
        # Initialize Groq using the key stored in settings
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        self._init_sqlite()

        # THE FULL GRAPH ONTOLOGY WITH DEFINITIONS
        self.RELATIONS_MAP = {
            # Identity & Demographics
            "HAS_NAME": "The user's actual name.",
            "HAS_NICKNAME": "The short name, handle, or alias the user prefers to be called by.",
            "HAS_CODE": "Strictly for numerical PINs, passwords, server access codes, or security keys. Extract the specific number or string provided, especially when the user is updating or replacing an old, compromised code.",
            "LIVES_IN": "Physical geography like cities or countries.",
            "SPEAKS_LANGUAGE": "Human languages the user speaks.",
            
            # Academic & Professional
            "STUDIES_AT": "The university or school the user attends.",
            "HAS_ROLE": "Professional job titles or specific project roles (e.g., Intern, Technical Lead).",
            "WORKS_AT": "Strictly for formal organizations, companies, research labs, or startups that employ or host the user. Do NOT use this for events, temporary competitions, or software projects.",
            
            # Technical & Capabilities
            "IS_A": "General descriptions of the user (e.g., engineering student, developer).",
            "USES_TOOL": "Software, frameworks, or programming languages (e.g., PyTorch, Python, C++).",
            "USES_OS": "Computer operating systems (e.g., Linux, Windows).",
            "HAS_SKILL": "Technical concepts or capabilities (e.g., MARL, backend engineering).",
            "WANTS_TO_LEARN": "Things the user explicitly states they want to study, improve, or achieve (e.g., English, IELTS).",
            
            # Projects & Ambitions
            "WORKS_ON": "For specific software projects, AI systems, digital architectures, or distinct events the user is actively building, developing, or organizing (e.g., a custom engine, an app).",
            "HAS_GOAL": "Long-term ambitions (e.g., getting a PhD, finding an internship).",
            
            # Personal Preferences
            "INTERESTED_IN": "Academic subjects, scientific fields, or deep research topics.",
            "HAS_HOBBIES": "Casual hobbies, sports, and games (e.g., chess, football)."
        }
        
        # Extract keys for the LLM prompt and database logic
        self.ALL_RELATIONS = list(self.RELATIONS_MAP.keys())
        
        # DYNAMIC OVERWRITE BOUNDS
        self.ONE_TO_ONE_RELATIONS = [
            "HAS_NAME", "HAS_CODE", "LIVES_IN", "STUDIES_AT", "HAS_ROLE", "USES_OS", "HAS_NICKNAME"
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
            f"2. You can ONLY use the relations defined in this exact schema map: {json.dumps(self.RELATIONS_MAP)}\n"
            "3. Read the definitions carefully to choose the most accurate relation.\n"
            "4. If no facts fit these relations, you MUST output {\"facts\": []}.\n"
            "5. RESTRAINT: Do NOT force extractions from casual conversation. If the user is just chatting, output {\"facts\": []}.\n"
            "6. IGNORE QUESTIONS: If the user is asking a question to retrieve information (e.g., 'What is my name?', 'Remind me my code'), DO NOT extract anything. Output {\"facts\": []}.\n\n"
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
    
    def get_relevant_facts(self, user_id: str, query_text: str, embedding_function, threshold: float = 0.6) -> str:
        """
        Retrieves semantic facts relevant to the current user prompt using vector similarity.
        """
        # Fetch ALL facts for this specific user
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT relation, object 
            FROM semantic_triplets 
            WHERE user_id = ?
        ''', (user_id,))
        
        facts = cursor.fetchall()
        
        if not facts:
            return ""

        # Vectorize the user's prompt once
        prompt_vector = embedding_function(query_text)
        
        relevant_facts = []

        for relation, obj in facts:
            # Combine relation with object for maximum semantic meaning
            # Replace underscores with spaces so the embedding model understands the words!
            clean_relation = relation.replace("_", " ")
            fact_string = f"{clean_relation} {obj}"
            # Vectorize the combined fact string
            fact_vector = embedding_function(fact_string)
            # Calculate Cosine Similarity (Math to see how close the vectors are)
            similarity = np.dot(prompt_vector, fact_vector) / (np.linalg.norm(prompt_vector) * np.linalg.norm(fact_vector))
            # If similarity > threshold, keep it!
            if similarity >= threshold:
                relevant_facts.append((similarity, relation, obj))

        # Return all the triplets that passed the threshold
        if not relevant_facts:
            return "" # None passed the threshold

        # Sort by highest similarity first (most relevant at the top)
        relevant_facts.sort(key=lambda x: x[0], reverse=True)

        # Format cleanly for the LLM prompt
        formatted_output = ["--- User Semantic Context ---"]
        for sim, rel, obj in relevant_facts:
            formatted_output.append(f"- {rel}: {obj}")
            print(f"[DEBUG] Retrieved: {rel} {obj} (Score: {sim:.2f})") # Uncomment to debug!
            
        return "\n".join(formatted_output)

# Instantiate the singleton
semantic_memory = SemanticMemory()