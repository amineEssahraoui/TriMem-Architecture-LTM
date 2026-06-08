import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.memory.procedural import procedural_memory
from app.core.embedding import get_embedding

# PHASE 1: EXTRACTION FOR MULTIPLE USERS
def run_multi_user_extraction():
    print("\n" + "="*80)
    print("PHASE 1 — MULTI-USER SOP EXTRACTION & SAVING")
    print("="*80)

    # We define 3 distinct users with conflicting rules to prove isolation
    test_data = {
        "user_amine_dev": [
            ("M1", "Always respond to me in French."),
            ("M2", "When writing code, never use classes. Only use functional programming."),
            ("M3", "Whenever I ask you to write an email, end it with 'Bien à toi, Amine'.")
        ],
        "user_sarah_research": [
            ("S1", "Always respond in English."),
            ("S2", "Always use bullet points for every response."),
            ("S3", "When you write code, strictly use Object-Oriented Programming (OOP) and classes."),
            ("S4", "When I share a research paper, always extract the mathematical formulas.")
        ],
        "user_alex_manager": [
            ("A1", "Habla siempre en español conmigo."),
            ("A2", "When asking for a sprint update, output a Markdown table with 'Task', 'Status', and 'Blocker'."),
            ("A3", "Keep all email drafts under 3 sentences.")
        ]
    }

    for user_id, messages in test_data.items():
        print(f"\n--- Extracting rules for {user_id.upper()} ---")
        for source_id, message in messages:
            print(f"[{source_id}] '{message}'")
            procedural_memory.extract_and_save_sop(
                user_id=user_id,
                text=message,
                source_id=source_id,
                embedding_function=get_embedding
            )

# PHASE 2: ISOLATION RETRIEVAL TEST
def run_multi_user_retrieval():
    print("\n" + "="*80)
    print("PHASE 2 — MULTI-USER ISOLATION RETRIEVAL TEST")
    print("="*80)

    # We ask the exact same questions to different users to see how the rules adapt
    queries = [
        # Coding Task (Amine vs Sarah should get conflicting rules)
        ("Write me a Python script to handle a database connection.", ["user_amine_dev", "user_sarah_research"]),
        
        # Email Task (Amine vs Alex should get conflicting rules)
        ("Can you draft an email to the client?", ["user_amine_dev", "user_alex_manager"]),

        # Research Task (Sarah should get research rules, Amine should only get his globals)
        ("Summarize this machine learning paper for me.", ["user_sarah_research", "user_amine_dev"]),

        # Sprint Task (Alex should get his table rule)
        ("Give me a sprint update based on these notes.", ["user_alex_manager"]),

        # General Knowledge (All users should ONLY get their global rules)
        ("What is the capital of Japan?", ["user_amine_dev", "user_sarah_research", "user_alex_manager"])
    ]

    for query, target_users in queries:
        print(f"\n{'─'*80}")
        print(f"QUERY: {query}")
        print(f"{'─'*80}")

        for user_id in target_users:
            print(f"\n>> Results for {user_id.upper()}:")
            result = procedural_memory.get_relevant_sops(
                user_id=user_id,
                query_text=query,
                embedding_function=get_embedding
            )
            
            if result:
                print(result)
            else:
                print("[No SOPs retrieved]")

# MAIN
if __name__ == "__main__":
    run_multi_user_extraction()
    run_multi_user_retrieval()
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)