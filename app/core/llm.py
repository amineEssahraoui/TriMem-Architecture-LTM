import ollama
import re
from app.core.config import settings

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
        "You are a strict Cognitive Memory Classifier for an AI memory database. "
        "Your ONLY job is to detect if a user is DECLARING new, storable facts about themselves. "
        "You must output a single integer from 1 to 10. No text, no markdown, no explanation."
    )

    prompt = (
        "Evaluate the Semantic Memory importance of the following user message on a scale of 1 to 10.\n\n"

        "CORE PRINCIPLE:\n"
        "- High scores (6-10) are STRICTLY reserved for DECLARATIVE statements where the user is sharing new facts, updating preferences, or establishing persistent identity traits.\n"
        "- Low scores (1-5) are for questions, commands, tasks, and temporary context.\n\n"

        "SCORING RUBRIC:\n"
        "- [1–3] NOISE & INQUIRY: Questions, retrieval requests, commands, coding tasks, greetings, or casual chat. "
        "(e.g., 'What is my code?', 'Write a Python script', 'Remind me my name', 'Hello!').\n"

        "- [4–5] EPHEMERAL: Temporary states, situational context, or current short-term actions. "
        "(e.g., 'I am fixing a bug today', 'I feel tired', 'I am reading a PDF right now').\n"

        "- [6–8] PROFILING: Stable user characteristics, interests, professions, habits, skills, goals, or long-term projects. "
        "(e.g., 'I am a software engineer', 'I like playing chess', 'I use Linux', 'My goal is IELTS 7.0').\n"

        "- [9–10] CRITICAL: Core identity, highly important personal facts, explicit persistent instructions, passwords, or hard rules. "
        "(e.g., 'My real name is David', 'My new vault code is 7777', 'Always answer in French').\n\n"

        "CRITICAL RULES:\n"
        "1. THE QUESTION TRAP: If the user is ASKING about their own data (e.g., 'What is my password?', 'Who am I?'), the score MUST BE 1. Do not confuse the importance of the topic with the intent of the message.\n"
        "2. THE TASK TRAP: Requests for the AI to do work (e.g., 'Fix this code', 'Write an email') MUST BE scored 1-3.\n"
        "3. HIGHEST ELEMENT: If a message contains both a task and a new fact (e.g., 'I am a doctor, write an email for me'), score based on the new fact (e.g., 7).\n\n"

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