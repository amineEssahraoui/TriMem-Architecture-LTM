from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from app.core.llm import generate_response, get_importance_score
from app.core.embedding import get_embedding
from app.memory import episodic_memory

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
    
    episodic_memory.add_interaction(
        user_id=user_id,
        user_message=user_message,
        ai_response=ai_response,
        embedding=embedding,
        importance_score=score
    )

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    try:
        query_embedding = get_embedding(request.message)
        if not query_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate embedding.")

        past_context = episodic_memory.retrieve_recent_context(
            user_id=request.user_id,
            query_embedding=query_embedding,
            query_text=request.message
        )

        if past_context:
            system_prompt = (
                "You are an intelligent, friendly, and helpful AI assistant.\n"
                "Use the following pieces of past memories to inform your response. "
                "If the memories are irrelevant, ignore them. Maintain consistency with past provided information.\n"
                "5. NO THINKING: DO NOT output any thinking process, internal monologue, or reasoning steps. Output ONLY your final conversational response directly.\n\n"
                f"=== PAST MEMORY ===\n{past_context}\n===================\n\n"
                "Respond naturally and conversationally to the user's latest message."
            )
        else:
            system_prompt = (
                "You are an intelligent, friendly, and helpful AI assistant. "
                "This is your very first interaction with this user. "
                "Your objective: Give a warm welcome, state clearly that you are ready to help, "
                "and keep your response brief but conversational.\n\n"
                "CRITICAL RULE: DO NOT output any thinking process, internal monologue, or reasoning steps. "
                "Output ONLY your final conversational response directly."
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