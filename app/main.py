from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.llm import generate_response
from app.embedding import get_embedding
from app.memory import memory_db

# Initialize the FastAPI application
app = FastAPI(title="TriMem Agent API - Phase 1")

# Define the expected data structure for an incoming chat request
class ChatRequest(BaseModel):
    user_id: str
    message: str

# Define the data structure for the API's response
class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Handles a user's chat message, retrieves context, and generates a response.
    """
    try:
        # 1. Generate an embedding for the incoming user message
        query_embedding = get_embedding(request.message)
        
        if not query_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate embedding from Ollama.")

        # 2. Retrieve relevant past context from episodic memory
        past_context = memory_db.retrieve_recent_context(
            user_id=request.user_id, 
            query_embedding=query_embedding
        )

        # 3. Construct the system prompt for the LLM, injecting the retrieved memories
        system_prompt = (
            "You are an intelligent and friendly AI assistant with an excellent memory. "
            "Below are excerpts from your previous conversations with this user.\n"
            "ABSOLUTE RULE: You MUST use this information to personalize your response "
            "(such as their name, preferences, etc.) and you must never contradict these facts.\n\n"
            f"=== CONVERSATION HISTORY ===\n{past_context}\n============================\n\n"
            "Using this context if relevant, respond naturally to the user's latest message."
        )

        # 4. Generate the response using Llama3
        ai_response = generate_response(prompt=request.message, system_prompt=system_prompt)

        # 5. Store this new interaction in the episodic memory for future use
        memory_db.add_interaction(
            user_id=request.user_id,
            user_message=request.message,
            ai_response=ai_response,
            embedding=query_embedding
        )

        return ChatResponse(response=ai_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))