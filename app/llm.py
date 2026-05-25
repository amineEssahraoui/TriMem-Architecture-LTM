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
        "You are an expert Cognitive Memory Classifier. "
        "Your task is to evaluate how important a user message is for long-term retention in an AI memory database. "
        "The more a message contains information that is important, persistent, or defining for the user, "
        "the higher the score must be. "
        "Your ONLY output must be a single integer from 1 to 10. "
        "No text, no explanation."
    )

    prompt = (
        "Evaluate the long-term memory importance of the following user message on a scale of 1 to 10.\n\n"

        "CORE PRINCIPLE:\n"
        "- The more the message contains information that is important to the user's identity, preferences, rules, "
        "long-term context, or future personalization value, the HIGHER the score.\n"
        "- Temporary, generic, task-specific, or low-signal information should receive LOWER scores.\n\n"

        "SCORING RUBRIC:\n"
        "- [1–3] NOISE: Greetings, small talk, emotions, temporary states, acknowledgments, or ephemeral context "
        "(e.g., 'Hello!', 'I am tired', 'Thanks').\n"

        "- [4–6] GENERAL: Questions, requests, short-term tasks, situational context, or factual discussions with "
        "limited long-term personalization value "
        "(e.g., 'How does Python work?', 'Fix this code', 'I am working on a report').\n"

        "- [7–8] PROFILING: Stable user characteristics, interests, professions, habits, skills, goals, "
        "long-term projects, or recurring preferences "
        "(e.g., 'I am a software engineer', 'I like jazz', 'I use Linux', 'I am learning machine learning').\n"

        "- [9–10] CRITICAL: Core identity, highly important personal facts, explicit persistent instructions, "
        "strong personalization rules, major constraints, or sensitive enduring information "
        "(e.g., 'My name is David', 'Always answer in French', 'I am allergic to peanuts').\n\n"

        "CRITICAL RULES:\n"
        "- If a message contains multiple types of information, ALWAYS score according to the HIGHEST-importance element.\n"
        "- Prefer scoring based on LONG-TERM usefulness and personalization value, not immediate task relevance.\n"
        "- Stable facts > temporary facts.\n"
        "- User identity, preferences, rules, and enduring context should strongly increase the score.\n\n"

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