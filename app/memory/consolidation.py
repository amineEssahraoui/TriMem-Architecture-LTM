import json
from groq import Groq
from app.core.config import settings
from app.memory.episodic import episodic_memory
from app.memory.semantic import semantic_memory

class MemoryConsolidation:
    def __init__(self):
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        self.N = settings.CONSOLIDATION_N  # trigger every N interactions

        # In-memory counter per user { user_id: count }
        self._counters: dict[str, int] = {}

    # PUBLIC TRIGGER
    def increment_and_check(self, user_id: str, embedding_function):
        """
        Called after every saved episodic interaction.
        Triggers consolidation when the counter reaches N.
        """
        self._counters[user_id] = self._counters.get(user_id, 0) + 1

        if self._counters[user_id] >= self.N:
            print(f"[Consolidation] Threshold reached for {user_id}. Starting consolidation...")
            self.consolidate(user_id, embedding_function)
            self._counters[user_id] = 0  # reset counter

    def consolidate(self, user_id: str, embedding_function):
        """
        Main consolidation pipeline:
        1. Fetch recent episodic interactions
        2. Ask Groq which are redundant vs semantic memory
        3. Delete redundant ones from ChromaDB
        4. Extract any new facts from the remaining ones
        """
        interactions = self._get_episodic_interactions(user_id)

        if not interactions:
            print(f"[Consolidation] No episodic interactions found for {user_id}.")
            return

        self._compress_and_clean(user_id, interactions, embedding_function)

    # STEP 1 — FETCH EPISODIC INTERACTIONS
    def _get_episodic_interactions(self, user_id: str) -> list[dict]:
        """
        Retrieves the N most recent episodic interactions for this user
        directly from ChromaDB.
        """
        try:
            results = episodic_memory.collection.get(
                where={"user_id": user_id},
                include=["documents", "metadatas"]
            )

            if not results or not results["ids"]:
                return []

            interactions = []
            for doc_id, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
                interactions.append({
                    "id": doc_id,
                    "content": doc,
                    "timestamp": float(meta.get("timestamp", 0)),
                    "importance_score": int(meta.get("importance_score", 5))
                })

            # Sort by most recent first, take N
            interactions.sort(key=lambda x: x["timestamp"], reverse=True)
            return interactions[:self.N]

        except Exception as e:
            print(f"[Consolidation] Failed to fetch episodic interactions: {e}")
            return []

    # STEP 2 — COMPRESS & CLEAN
    def _compress_and_clean(self, user_id: str, interactions: list[dict], embedding_function):
        """
        Sends episodic interactions to Groq.
        Groq identifies which are redundant given existing semantic facts.
        Deletes redundant ones. Extracts new facts from the rest.
        """
        # Get current semantic facts for this user
        cursor = semantic_memory.conn.cursor()
        cursor.execute(
            "SELECT relation, object FROM semantic_triplets WHERE user_id = ?",
            (user_id,)
        )
        existing_facts = cursor.fetchall()
        facts_summary = ", ".join([f"{r}: {o}" for r, o in existing_facts]) or "None"

        # Format interactions for Groq
        interactions_text = "\n".join([
            f"[ID: {i['id']}] {i['content']}"
            for i in interactions
        ])

        system_prompt = (
            "You are a Memory Consolidation Engine for an AI agent.\n"
            "Your job is to analyze a list of past episodic interactions and decide which ones "
            "are already fully captured by the existing semantic facts.\n\n"

            "RULES:\n"
            "1. An interaction is REDUNDANT if ALL its meaningful information already exists in the semantic facts.\n"
            "2. An interaction is VALUABLE if it contains context, nuance, or information NOT in the semantic facts.\n"
            "3. Be aggressive about redundancy — if the core fact is already stored, the interaction is redundant.\n\n"
            "4. Be VERY aggressive about redundancy — simple one-fact interactions like "
            "'My name is X' or 'I use Y' are ALWAYS redundant if the fact exists in semantic memory.\n"

            "OUTPUT FORMAT: Strict JSON only. No preamble.\n"
            "{\n"
            '  "redundant_ids": ["id1", "id2"],\n'
            '  "valuable_interactions": ["full content of valuable interaction 1", ...]\n'
            "}"
        )

        prompt = (
            f"EXISTING SEMANTIC FACTS:\n{facts_summary}\n\n"
            f"EPISODIC INTERACTIONS TO ANALYZE:\n{interactions_text}\n\n"
            "Identify which interactions are redundant and which are still valuable."
        )

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )

            parsed = json.loads(response.choices[0].message.content)
            redundant_ids = parsed.get("redundant_ids", [])
            valuable_interactions = parsed.get("valuable_interactions", [])

            # DELETE redundant interactions from ChromaDB
            if redundant_ids:
                episodic_memory.collection.delete(ids=redundant_ids)
                print(f"[Consolidation] Deleted {len(redundant_ids)} redundant episodic interactions.")

            # EXTRACT new facts from valuable interactions not yet in semantic memory
            for content in valuable_interactions:
                semantic_memory.extract_and_save_facts(
                    user_id=user_id,
                    text=content,
                    source_id=f"consolidation_{user_id}"
                )

            print(f"[Consolidation] Done. {len(redundant_ids)} deleted, {len(valuable_interactions)} re-processed.")

        except Exception as e:
            print(f"[Consolidation] Groq compression failed: {e}")


# Singleton
memory_consolidation = MemoryConsolidation()