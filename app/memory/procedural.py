import chromadb
import os
import json
from groq import Groq
from app.core.embedding import get_embedding
from app.core.config import settings

class ProceduralMemory:
    def __init__(self):
        # Shared ChromaDB client — same path as episodic memory
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name="procedural_memory",
            metadata={"hnsw:space": "cosine"}
        )

        # Groq client for SOP extraction
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)

        # Universal SOPs are stored with this reserved user_id
        self.UNIVERSAL_USER_ID = "universal"

        # Load and seed universal SOPs from JSON on startup
        self._seed_universal_sops()

    # Universal SOPs (loaded from JSON)
    def _seed_universal_sops(self):
        """
        Loads universal SOPs from the JSON file and seeds ChromaDB.
        Uses the SOP 'id' field to avoid duplicate insertions.
        """
        sop_file = settings.UNIVERSAL_SOPS_PATH

        if not os.path.exists(sop_file):
            print(f"[Procedural] No universal SOPs file found at: {sop_file}")
            return

        with open(sop_file, "r", encoding="utf-8") as f:
            sops = json.load(f)

        for sop in sops:
            sop_id = sop.get("id")
            instruction = sop.get("instruction", "")
            task_tag = sop.get("task_tag", None)

            if not sop_id or not instruction:
                print(f"[Procedural] Skipping malformed universal SOP: {sop}")
                continue

            # Check if this SOP already exists to avoid re-seeding on every restart
            existing = self.collection.get(ids=[sop_id])
            if existing and existing["ids"]:
                continue

            # Import embedding function here to avoid circular imports
            embedding = get_embedding(instruction)

            if not embedding:
                print(f"[Procedural] Failed to embed universal SOP: {sop_id}")
                continue

            self.collection.add(
                ids=[sop_id],
                embeddings=[embedding],
                documents=[instruction],
                metadatas=[{
                    "user_id": self.UNIVERSAL_USER_ID,
                    "scope": "universal",
                    "task_tag": task_tag or ""
                }]
            )
            print(f"[Procedural] Seeded universal SOP: {sop_id}")

    # Personal SOPs from user messages via Groq
    def extract_and_save_sop(self, user_id: str, text: str, source_id: str, embedding_function):
        """
        Uses Groq to detect if a user message contains a behavioral instruction.
        If found, classifies it and saves it to ChromaDB.
        """
        system_prompt = (
            "You are an expert Cognitive Memory Extractor for an autonomous AI agent.\n"
            "Your objective is to analyze the user's message and extract PERSISTENT PROCEDURAL INSTRUCTIONS (SOPs).\n"
            "A persistent instruction is a behavioral rule or formatting preference the user wants the AI to strictly follow in all future interactions.\n\n"

            "--- CLASSIFICATION RULES ---\n"
            "1. SCOPE (CRITICAL):\n"
            "   - 'global': UNCONDITIONAL RULES. The rule applies universally to EVERY SINGLE interaction, regardless of the topic (e.g., 'Always speak French', 'Be concise', 'Never use emojis').\n"
            "   - 'task_specific': CONDITIONAL RULES. The rule ONLY activates under a specific condition, when performing a specific task, or discussing a specific topic (e.g., 'When writing code...', 'If drafting an email...', 'For research papers...').\n"
            "   -> STRICT RULE: If the user's instruction contains \"when...\", \"if...\", or targets a specific activity (code, emails, math, summaries), it MUST be classified as 'task_specific'.\n\n"

            "2. TASK TAG:\n"
            "   - If 'task_specific', generate a broad, generic 1-2 word category tag (e.g., 'programming', 'email_writing', 'research_analysis', 'project_management').\n"
            "   - If 'global', this field MUST be exactly null.\n\n"

            "3. EXCLUSIONS (Output {\"found\": false}):\n"
            "   - Casual chat or greetings.\n"
            "   - Ephemeral one-time tasks ('Fix this specific bug', 'Draft an email for me today').\n"
            "   - Identity facts ('I live in Paris', 'I am an engineer').\n\n"

            "--- EXAMPLES ---\n"
            "User: 'From now on, always speak to me in French.'\n"
            "Output: {\"found\": true, \"instruction\": \"Always respond in French.\", \"scope\": \"global\", \"task_tag\": null}\n\n"

            "User: 'When you write code, strictly use Object-Oriented Programming (OOP) and classes.'\n"
            "Output: {\"found\": true, \"instruction\": \"Always use OOP and classes when writing code.\", \"scope\": \"task_specific\", \"task_tag\": \"programming\"}\n\n"

            "User: 'Keep all email drafts under 3 sentences.'\n"
            "Output: {\"found\": true, \"instruction\": \"Keep email drafts under 3 sentences.\", \"scope\": \"task_specific\", \"task_tag\": \"email_writing\"}\n\n"

            "--- OUTPUT FORMAT ---\n"
            "Return ONLY a valid JSON object. No preamble, no markdown formatting. Schema:\n"
            "{\"found\": boolean, \"instruction\": string, \"scope\": \"global\" | \"task_specific\", \"task_tag\": string | null}"
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

            parsed = json.loads(response.choices[0].message.content)

            if not parsed.get("found", False):
                print(f"[Procedural] No SOP detected in message.")
                return

            instruction = parsed.get("instruction", "").strip()
            scope = parsed.get("scope", "global")
            task_tag = parsed.get("task_tag")
            if task_tag is None:
                task_tag = ""

            if not instruction:
                return

            # Build a unique SOP ID for this user + instruction
            sop_id = f"{user_id}_sop_{source_id}"

            if scope == "task_specific" and task_tag:
                clean_tag = task_tag.replace('_', ' ')
                searchable_document = f"Task category: {clean_tag}. Action to apply: {instruction}"
            else:
                searchable_document = instruction

            # Embed the instruction text (using the enriched document)
            embedding = embedding_function(searchable_document)
            if not embedding:
                print(f"[Procedural] Failed to embed personal SOP.")
                return

            # Upsert: overwrite if SOP with same ID already exists
            self.collection.upsert(
                ids=[sop_id],
                embeddings=[embedding],
                documents=[searchable_document], # On sauvegarde le texte enrichi
                metadatas=[{
                    "user_id": user_id,
                    "scope": scope,
                    "task_tag": task_tag
                }]
            )
            print(f"[Procedural] SAVED personal SOP [{scope}|{task_tag or 'global'}]: {instruction}")

        except Exception as e:
            print(f"[Procedural Error] SOP extraction failed: {e}")

    # Hierarchical: Global → Specific → Universal
    def get_relevant_sops(self, user_id: str, query_text: str, embedding_function, n_results: int = 3) -> str:
        """
        Retrieves the most relevant SOPs for the current query.
        Hierarchy: Personal Global (All) → Personal Specific (Vector Match) → Universal (Vector Match)
        """
        query_embedding = embedding_function(query_text)
        sections = []

        # Personal Global SOPs (Fetch them all directly)
        try:
            global_results = self.collection.get(
                where={"$and": [{"user_id": {"$eq": user_id}}, {"scope": {"$eq": "global"}}]}
            )
            personal_global = global_results.get("documents", [])
            if personal_global:
                sections.append("=== Your Preferences (Always Apply) ===")
                sections.extend([f"- {doc}" for doc in personal_global])
        except Exception as e:
            print(f"[Procedural] Global fetch warning: {e}")

        # Personal Task-Specific SOPs (Vector Search)
        personal_specific = self._query_collection(
            query_embedding=query_embedding,
            user_id=user_id,
            scope_filter="task_specific",
            n_results=n_results
        )
        if personal_specific:
            sections.append("=== Task-Specific Rules (Apply When Relevant) ===")
            sections.extend([f"- {doc}" for doc in personal_specific])

        # Universal SOPs (Vector Search)
        universal = self._query_collection(
            query_embedding=query_embedding,
            user_id=self.UNIVERSAL_USER_ID,
            scope_filter="universal", # Ensure it only looks for universal scope
            n_results=n_results
        )
        if universal:
            sections.append("=== System Guidelines (Base Rules) ===")
            sections.extend([f"- {doc}" for doc in universal])

        if not sections:
            return ""

        return "\n".join(sections)

    def _query_collection(self, query_embedding: list, user_id: str, scope_filter: str | None, n_results: int) -> list[str]:
        """
        Internal helper: queries ChromaDB with user_id and optional scope filter.
        Returns a list of document strings.
        """
        try:
            # Build the where clause
            if scope_filter:
                where = {
                    "$and": [
                        {"user_id": {"$eq": user_id}},
                        {"scope": {"$eq": scope_filter}}
                    ]
                }
            else:
                where = {"user_id": {"$eq": user_id}}

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "distances"]
            )

            docs = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]

            # Filter by similarity threshold (distance in cosine space)
            filtered = [
                doc for doc, dist in zip(docs, distances)
                if dist < 0.60
            ]

            return filtered

        except Exception as e:
            print(f"[Procedural] Query warning: {e}")
            return []


# Singleton instance
procedural_memory = ProceduralMemory()