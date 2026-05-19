import ollama
import re
from app.config import settings

def generate_response(prompt : str, system_prompt: str = "") -> str: # Empty System_prompt by default !
    """
    Sends a prompt to the LLM and returns its generated response.
    """
    messages = []
    if system_prompt: 
        messages.append({"role": "system", "content" : system_prompt}) # Add system prompte if available

    messages.append({"role": "user", "content": prompt}) # Add user prompte

    try: 
        response = ollama.chat(
            model = settings.LLM_MODEL,
            messages = messages
        )
        return response['message']['content']
    except Exception as e: 
        print(f"Error communicating with Ollama: {e}")
        return "Sorry, I am having trouble connecting to my language model."
    
def get_importance_score(message: str) -> int:
    """
    Asks a lightweight LLM (e.g., TinyLlama) to evaluate the importance of the message.
    """
    system_prompt = "You are a strict data classification system. You output ONLY a single integer."
    
    prompt = (
        "Rate the importance of the following user message for a long-term AI memory system on a scale of 1 to 10.\n"
        "- 1 to 3: Meaningless chatter, greetings, or short acknowledgments (e.g., 'hello', 'okay', 'thanks').\n"
        "- 4 to 7: General questions or context (e.g., 'how does python work?').\n"
        "- 8 to 10: Highly important personal facts, names, preferences, or secrets (e.g., 'My name is Alice', 'I am allergic to peanuts').\n\n"
        "ABSOLUTE RULE: Output ONLY a single integer between 1 and 10. Do not write any text or explanation.\n\n"
        f"User Message: '{message}'"
    )
    
    try:
        # We explicitly use the tiny model here for speed!
        response = ollama.chat(
            model="tinyllama", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        
        output_text = response['message']['content']
        
        # Regex to extract the first number found in the output (in case the LLM disobeys and writes text)
        numbers = re.findall(r'\d+', output_text)
        if numbers:
            score = int(numbers[0])
            # Ensure the score stays within the 1-10 bounds
            return max(1, min(10, score))
            
        return 5 # Default fallback
        
    except Exception as e:
        print(f"Scoring error: {e}")
        return 5 # Default fallback if Ollama crashes