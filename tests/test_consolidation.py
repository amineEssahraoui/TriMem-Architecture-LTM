import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.memory.episodic import episodic_memory
from app.memory.consolidation import memory_consolidation
from app.core.embedding import get_embedding

TEST_USER = "test_user_consolidation"

# 1 — Seed episodic interactions manually
def seed_episodic():
    print("\n" + "="*60)
    print("1 — Seeding Episodic Interactions")
    print("="*60)

    interactions = [
        # Redundant ones (facts already simple enough to be in semantic)
        ("I use Linux as my main OS.", "Got it, I'll remember that."),
        ("My name is Amine.", "Nice to meet you, Amine!"),
        ("I am building a project called TriMem.", "That sounds interesting!"),

        # Valuable ones (context that goes beyond a simple fact)
        ("I use Linux but I'm having trouble with ChromaDB permissions on my machine.", "Let me help you debug that."),
        ("TriMem is a hybrid memory architecture combining episodic, semantic and procedural memory.", "That's a fascinating architecture."),
    ]

    for user_msg, ai_msg in interactions:
        embedding = get_embedding(user_msg)
        episodic_memory.add_interaction(
            user_id=TEST_USER,
            user_message=user_msg,
            ai_response=ai_msg,
            embedding=embedding,
            importance_score=7
        )
        print(f"[Seeded] {user_msg[:60]}")


# 2 — Check before consolidation
def check_before():
    print("\n" + "="*60)
    print("2 — Episodic Count BEFORE Consolidation")
    print("="*60)

    results = episodic_memory.collection.get(where={"user_id": TEST_USER})
    print(f"Total episodic interactions: {len(results['ids'])}")
    for doc_id, doc in zip(results["ids"], results["documents"]):
        print(f"  [{doc_id}] {doc[:80]}")


# 3 — Run consolidation
def run_consolidation():
    print("\n" + "="*60)
    print("3 — Running Consolidation")
    print("="*60)

    memory_consolidation.consolidate(TEST_USER, embedding_function=get_embedding)


# 4 — Check after consolidation
def check_after():
    print("\n" + "="*60)
    print("4 — Episodic Count AFTER Consolidation")
    print("="*60)

    results = episodic_memory.collection.get(where={"user_id": TEST_USER})
    print(f"Total episodic interactions remaining: {len(results['ids'])}")
    for doc_id, doc in zip(results["ids"], results["documents"]):
        print(f"  [{doc_id}] {doc[:80]}")


# MAIN
if __name__ == "__main__":
    seed_episodic()
    check_before()
    run_consolidation()
    check_after()

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)