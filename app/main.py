from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from app.llm import generate_response, get_importance_score
from app.embedding import get_embedding
from app.memory import memory_db

app = FastAPI(title="TriMem Agent API - Phase 1")

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

def save_memory_background(user_id: str, user_message: str, ai_response: str, embedding: list[float]):
    """Calculates importance score and saves memory without blocking the API response."""
    
    score = get_importance_score(user_message)
    print(f"[Background Task] Importance Score for '{user_id}': {score}/10")
    
    memory_db.add_interaction(
        user_id=user_id,
        user_message=user_message,
        ai_response=ai_response,
        embedding=embedding,
        importance_score=score
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    try:
        query_embedding = get_embedding(request.message)
        if not query_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate embedding.")

        past_context = memory_db.retrieve_recent_context(
            user_id=request.user_id, 
            query_embedding=query_embedding,
            query_text=request.message
        )

        # Dynamic System Prompt
        if past_context.strip():
            # If the agent found memories in the database (Returning User)
            system_prompt = (
                "You are an intelligent, friendly, and highly efficient AI assistant. "
                "Below are facts from your previous conversations with this user.\n\n"
                "RULES:\n"
                "1. CONCISENESS: Be direct and to the point. Do not use excessive filler words or unnecessary pleasantries unless strictly needed.\n"
                "2. PERSONALIZATION: You MUST use the past memory to personalize your response if relevant.\n"
                "3. NATURAL TONE: Do not explicitly say 'I remember from our previous conversation' or 'Based on my memory'. Act naturally.\n\n"
                f"=== PAST MEMORY ===\n{past_context}\n===================\n\n"
                "Respond naturally and concisely to the user's latest message."
            )
        else:
            # If this is the very first time they speak (Empty memory)
            system_prompt = (
                "You are an intelligent, friendly, and highly efficient AI assistant. "
                "This is your very first interaction with this user. "
                "Your objective: Give a quick, warm welcome, state clearly that you are ready to help, "
                "and keep your response brief and to the point. Do not be overly verbose."
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