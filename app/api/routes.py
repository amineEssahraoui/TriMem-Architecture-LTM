import time
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from app.core.llm import generate_response, get_importance_score
from app.core.embedding import get_embedding
from app.memory.episodic import episodic_memory
from app.memory.semantic import semantic_memory
from app.memory.procedural import procedural_memory

router = APIRouter()

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

def save_memory_background(user_id: str, user_message: str, ai_response: str, embedding: list[float]):
    """Calculates importance score and saves memory without blocking the API response."""
    score = get_importance_score(user_message)
    print(f"[Background Task] Importance Score for '{user_id}': {score}/10")

    timestamp = str(time.time())
    memory_id = f"{user_id}_{timestamp}"
    
    episodic_memory.add_interaction(
        user_id=user_id,
        user_message=user_message,
        ai_response=ai_response,
        embedding=embedding,
        importance_score=score
    )

    if score >= 6:
        print(f"[Background Task] Score {score} >= 6. Triggering Semantic Graph Pipeline...")

        semantic_memory.extract_and_save_facts(
            user_id=user_id,
            text=user_message,
            source_id=memory_id
        )

        procedural_memory.extract_and_save_sop(
            user_id=user_id,
            text=user_message,
            source_id=memory_id,
            embedding_function=get_embedding
        )

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    try:
        query_embedding = get_embedding(request.message)
        if not query_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate embedding.")

        episodic_context = episodic_memory.retrieve_recent_context(
            user_id=request.user_id,
            query_embedding=query_embedding,
            query_text=request.message
        )

        semantic_context = semantic_memory.get_relevant_facts(
            user_id=request.user_id, 
            query_text=request.message,
            embedding_function=get_embedding,
            threshold=0.5
        )

        procedural_context = procedural_memory.get_relevant_sops(
            user_id=request.user_id,
            query_text=request.message,
            embedding_function=get_embedding
        )

        if episodic_context or semantic_context:
            system_prompt = (
                "You are TriMem, an intelligent AI assistant with a persistent multi-layered memory system.\n"
                "You remember who the user is, what they know, and how they want to be helped.\n\n"

                "RULES:\n"
                "1. GROUND TRUTH OVERRIDE: The 'FACTS CONTEXT' (Semantic Memory) is the absolute, most up-to-date truth. "
                "If there is a conflict between past conversations and known facts (e.g., an updated vault code, changed OS, new role), ALWAYS trust the FACTS CONTEXT.\n"
                "2. PROCEDURAL FIRST: Before formulating your response, read the BEHAVIORAL RULES. "
                "If a rule applies to this query, follow it unconditionally. User personal rules override system guidelines.\n"
                "3. RELEVANCE: Only use memory when it genuinely adds value. If memory is irrelevant to the question, simply answer the question.\n"
                "4. NO INVENTION: Do NOT fabricate memories, facts, or user information. If you do not know something, say so.\n"
                "5. USER ISOLATION: You are talking to ONE user. Never reference or reveal data from other users.\n"
                "6. TONE: Be warm, direct, and professional. Speak like a knowledgeable colleague, not a customer service bot. No excessive emojis.\n"
                "7. NO THINKING: DO NOT output reasoning steps, internal monologue, or chain-of-thought. Output ONLY your final response.\n\n"

                f"=== PAST CONTEXT (Episodic Memory — Past Conversations) ===\n"
                f"{episodic_context}\n"
                f"===========================================================\n\n"

                f"=== FACTS CONTEXT (Semantic Memory — Absolute Ground Truth) ===\n"
                f"{semantic_context}\n"
                f"================================================================\n\n"

                f"=== BEHAVIORAL RULES (Procedural Memory — Follow Unconditionally) ===\n"
                f"{procedural_context}\n"
                f"======================================================================\n\n"

                "Respond naturally to the user's latest message."
            )
        else:
            system_prompt = (
                "You are an intelligent, natural, and helpful AI assistant.\n"
                "This is your very first interaction with this user. There is no prior memory.\n\n"
                "Your objective: Give a warm, natural welcome, state clearly that you are ready to help, "
                "and keep your response brief but conversational.\n\n"
                "RULES:\n"
                "1. TONE: Be warm and professional. Do NOT use excessive emojis. Speak like a helpful human colleague, not an overly eager customer service bot.\n"
                "2. NO THINKING: DO NOT output any thinking process, internal monologue, or reasoning steps. Output ONLY your final conversational response directly."
            )

        ai_response = generate_response(prompt=request.message, system_prompt=system_prompt)

        background_tasks.add_task(
            save_memory_background,
            user_id=request.user_id,
            user_message=request.message,
            ai_response=ai_response,
            embedding=query_embedding
        )

        return ChatResponse(response=ai_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))