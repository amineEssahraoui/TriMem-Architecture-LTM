import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.llm import get_importance_score

if __name__ == "__main__":
    print("=== STARTING SCORING MODEL TEST ===")
    
    # Define test cases: [Category, Message]
    test_cases = [
        # --- [1 to 3] NOISE & TRIVIAL CHATTER ---
        ("Trivial", "Hello!"),
        ("Trivial", "Okay, thanks for the help."),
        ("Trivial", "Good morning, how are you doing?"),
        ("Trivial", "I am feeling a bit tired today."),
        ("Trivial", "Wow, that is really cool."),
        ("Trivial", "See you later!"),
        ("Trivial", "Yup, exactly what I thought."),

        # --- [4 to 6] GENERAL & CONTEXTUAL ---
        ("General", "Can you explain how a database works?"),
        ("General", "What is the capital of Japan?"),
        ("General", "Write a python script to parse a JSON file."),
        ("General", "Translate the following sentence to Spanish please."),
        ("General", "How long does it take to boil an egg?"),
        ("General", "Can you summarize the plot of the movie Inception?"),
        ("General", "What is the weather usually like in Paris during May?"),

        # --- [7 to 8] PROFILING & PREFERENCES ---
        ("Profiling", "I really enjoy playing the electric guitar on weekends."),
        ("Profiling", "I usually drink my coffee black with no sugar."),
        ("Profiling", "My favorite movie director is Christopher Nolan."),
        ("Profiling", "I work as a senior data analyst in a bank."),
        ("Profiling", "I always prefer using dark mode on all my IDEs and apps."),
        ("Profiling", "I drive a 2018 Honda Civic."),
        ("Profiling", "I am currently trying to learn how to speak Japanese."),

        # --- [9 to 10] CRITICAL & IDENTITY (Highest Value Rule triggers) ---
        ("Critical", "Never forget that my secret pin code is 9876."),
        ("Critical", "I am severely allergic to seafood and peanuts."),
        ("Critical", "My name is David and I work as a software engineer."), # Mixed: Name (10) + Job (7)
        ("Critical", "My wife's name is Sarah and her birthday is October 12th."),
        ("Critical", "Always address me as 'Captain' in our future conversations."),
        ("Critical", "I have type 1 diabetes, so keep that in mind for any diet suggestions."),
        ("Critical", "Hello there! Just so you know, my home address is 123 Evergreen Terrace.") # Mixed: Greeting (1) + Address (9)
    ]
    
    for category, message in test_cases:
        print("-" * 50)
        print(f"Testing [{category}] message: '{message}'")
        
        start_time = time.time()
        # Calling the function directly from your app.llm module
        score = get_importance_score(message)
        execution_time = time.time() - start_time
        
        print(f"Result : Score = {score}/10")
        print(f"Time   : {execution_time:.2f} seconds")

    print("-" * 50)
    print("=== TEST COMPLETED ===")