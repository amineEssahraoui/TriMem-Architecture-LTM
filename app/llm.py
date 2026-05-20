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
    Asks a lightweight LLM to evaluate the importance of the message.
    """
    system_prompt = (
        "You are an expert Cognitive Memory Classifier. Your task is to evaluate the importance of "
        "retaining a user message in a long-term AI memory database. "
        "Your ONLY output must be a single integer. No text, no explanation."
    )
    
    prompt = (
        "Evaluate the long-term retention importance of the following user message on a scale of 1 to 10.\n\n"
        "SCORING RUBRIC:\n"
        "- [1 to 3] NOISE: Greetings, small talk, pleasantries, or temporary states (e.g., 'Hello!', 'I am tired', 'Thanks').\n"
        "- [4 to 6] GENERAL: Factual questions, tasks, or situational context (e.g., 'How does Python work?', 'Fix this code').\n"
        "- [7 to 8] PROFILING: Long-term traits, professions, hobbies, or general preferences (e.g., 'I am a software engineer', 'I like jazz').\n"
        "- [9 to 10] CRITICAL: Core identity (the user's NAME), severe health facts (allergies), secrets, or explicit user rules (e.g., 'My name is David', 'My PIN is 1234').\n\n"
        "CRITICAL RULE: If a message contains a mix of information (e.g., a greeting AND a name, or a profession AND a name), "
        "you MUST base your final score on the HIGHEST value element present.\n\n"
        f"User Message: '{message}'\n\n"
        "OUTPUT FORMAT: Return ONLY the integer."
    )
    try:
        # We explicitly use the tiny model here for speed!
        response = ollama.chat(
            model=settings.SCORING_MODEL, 
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