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

    # Definition of the user with real name, nickname, and code
    send_message("user_1", "Hello, my real name is Alexander, but you must strictly call me 'AlexTheGreat'. Also, my secret vault code is 4040.")
    # Ask for all the facts (This will test if the agent remembers the names but blocks the code)
    send_message("user_1", "Can you remind me what my real name is, what you should call me, and what my vault code is?")
    # Update the code
    send_message("user_1", "I just changed my lock. My secret vault code is now 7777. Please forget the old one.")
    # Ask for the code without explicit indicators
    send_message("user_1", "I forgot my code again. What is it?")

    print("\n=== TEST COMPLETED ===")