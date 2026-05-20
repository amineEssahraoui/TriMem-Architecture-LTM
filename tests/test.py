import urllib.request
import json
import time

# The URL of your local FastAPI server
API_URL = "http://127.0.0.1:8000/chat"

def send_message(user_id: str, message: str, wait_time: int = 3):
    """
    Sends a POST request to the API, prints the interaction, 
    and waits for background tasks to complete.
    """
    print(f"\n[{user_id.upper()}] : {message}")
    
    # Prepare the data as JSON
    data = json.dumps({"user_id": user_id, "message": message}).encode("utf-8")
    req = urllib.request.Request(API_URL, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        # Send the request and read the response
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            print(f"[AGENT] : {result['response']}")
    except Exception as e:
        print(f"Error connecting to the server: {e}")
        
    # Wait for the BackgroundTask (Scoring + ChromaDB save) to finish
    print(f"(Waiting {wait_time}s for background memory consolidation...)")
    time.sleep(wait_time)

if __name__ == "__main__":
    print("=== STARTING GLOBAL MEMORY INTEGRATION TEST ===")

    # TEST 1: CORE IDENTITY & NICKNAME (USER 1)
    send_message("user_1", "Hello! My real name is Alexander, but please only call me 'AlexTheGreat'. Also, my secret vault code is 4040.")
    
    # TEST 2: PREFERENCES & ALLERGIES (USER 2)
    send_message("user_2", "Hi, I am Sarah. I love Italian food, but I am extremely allergic to garlic.")

    # TEST 3: RECALL & NICKNAME ENFORCEMENT (USER 1)
    send_message("user_1", "I forgot my vault code, can you remind me? And what should you call me?")

    # TEST 4: MEMORY ISOLATION / CROSS-CONTAMINATION (USER 2)
    send_message("user_2", "I am hungry. Can you suggest a dinner idea based on what you know about me? Also, do you know any vault codes?")
    
    # TEST 5: TRIVIAL CHATTER (NOISE FILTERING) (USER 1)
    send_message("user_1", "Wow, it is really raining hard outside today.")
    
    # TEST 6: FINAL CHECK (USER 1)
    send_message("user_1", "Just checking, do you remember my full real name and my nickname?")

    print("\n=== TEST COMPLETED ===")