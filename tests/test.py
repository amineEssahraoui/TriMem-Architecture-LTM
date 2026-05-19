import urllib.request
import json

# The URL of your local FastAPI server
API_URL = "http://127.0.0.1:8000/chat"

def send_message(user_id: str, message: str):
    """
    Sends a POST request to the API and prints the interaction.
    """
    print(f"\[{user_id}] : {message}")
    
    # Prepare the data as JSON
    data = json.dumps({"user_id": user_id, "message": message}).encode("utf-8")
    req = urllib.request.Request(API_URL, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        # Send the request and read the response
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            print(f"[Agent] : {result['response']}")
    except Exception as e:
        print(f"Error connecting to the server: {e}")

if __name__ == "__main__":
    print("=== STARTING CROSS-MEMORY TEST ===")

    # Feed information to User 1
    send_message("user_1", "Bonjour, je suis Alice et mon fruit préféré est la pomme.")

    # Feed information to User 2
    send_message("user_2", "Salut, je m'appelle Bob et j'adore les bananes.")

    # Test memory isolation for User 1
    send_message("user_1", "Peux-tu me rappeler mon prénom et mon fruit préféré ?")

    # Test memory isolation for User 2
    send_message("user_2", "Et moi, comment je m'appelle et quel fruit est-ce que j'aime ?")
    
    print("\n=== TEST COMPLETED ===")