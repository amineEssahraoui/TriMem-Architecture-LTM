import ollama
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