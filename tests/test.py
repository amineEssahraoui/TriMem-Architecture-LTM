import urllib.request
import json
import time

# The URL of your local FastAPI server
API_URL = "http://127.0.0.1:8000/chat"

def send_message(user_id: str, message: str, wait_time: int = 120):
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

    # User 1 establishes baseline identity
    send_message("user_1", "Hey, I'm Amine. I strictly use Linux, I'm building TriMem, and my secret vault code is 8080.")
    
    # User 2 establishes a conflicting identity
    send_message("user_2", "Hi, I'm Sarah. I use macOS for design.")
    
    # User 3 attempts a Task + Fact combination (Dual-Intent Trap)
    # CRITICAL: The AI must score this high enough to save 'mathematician', but ignore the task part.
    send_message("user_3", "I am a mathematician. Can you write a quick Python function to calculate a factorial? Just the code please.")
    
    # User 1 mixes pure noise with a new fact.
    # CRITICAL: Must ignore the pizza, but extract 'Python'.
    send_message("user_1", "Man, I'm starving, I might order pizza. Anyway, my favorite programming language is Python.")
    
    # User 2 uses a hypothetical statement.
    # CRITICAL: The AI must NOT extract 'Windows'. It must realize this is a joke/hypothetical.
    send_message("user_2", "Imagine if I used Windows haha, that would be terrible. Glad I stick to macOS.")

    
    # User 2 tries to probe User 1's data.
    # CRITICAL: Must deny knowledge of Amine or his code.
    send_message("user_2", "Do you know if Amine has a vault code set up?")
    
    # User 3 asks if the agent has a code for them.
    # CRITICAL: Must gracefully say no, without hallucinating one or bleeding User 1's code.
    send_message("user_3", "Do you know if I have a vault code saved in my file?")
    
    # User 1 tests awareness without explicitly asking for the value.
    # CRITICAL: The Hybrid Search MUST trigger on the word 'code' and confirm it knows the code is 8080.
    send_message("user_1", "Hey, do you happen to know if my secret vault code is safely stored?")
    
    # User 1 updates their code.
    send_message("user_1", "I just changed my lock. The new vault code is 9999. Forget the old 8080 one.")
    
    # User 2 updates their OS.
    send_message("user_2", "I actually switched my main OS from macOS to Ubuntu Linux today.")
    
    # User 1 asks a multi-part question hitting both new and old facts.
    # CRITICAL: Must retrieve Linux, Python, and the UPDATED 9999 code.
    send_message("user_1", "Quick check before I log off: what is my favorite programming language, and what is my vault code?")
    
    # User 2 checks the OS update.
    # CRITICAL: Must say Ubuntu/Linux, NOT macOS.
    send_message("user_2", "Remind me, what OS do I use now?")
    
    # User 3 checks the Dual-Intent extraction from Phase 1.
    # CRITICAL: Must remember the profession, proving the factorial task didn't ruin the fact extraction.
    send_message("user_3", "What is my profession?")

    print("\n=== TEST COMPLETED ===")