import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from typing import List, Any
import json
from datetime import datetime, timezone

# Import the configured S3 client and bucket name from our config file
from r2_config import s3_client, R2_BUCKET_NAME

# --- FastAPI App Initialization ---
app = FastAPI(
    title="PythonChatMicroservice",
    description="An API that uses Groq for chat and saves conversations to Cloudflare R2.",
    version="1.2.0",
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Groq Model Initialization ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set!")

try:
    chat_model = ChatGroq(
        temperature=0.7,
        groq_api_key=GROQ_API_KEY,
        model_name="gemma2-9b-it",
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize ChatGroq model: {e}")


# --- Pydantic Models for Data Validation ---
class PromptRequest(BaseModel):
    text: str = Field(..., min_length=1)

class LLMResponse(BaseModel):
    response: str

class ChatMessage(BaseModel):
    id: Any
    text: str
    sender: str

# This defines the expected request body for saving a chat
class ChatHistoryRequest(BaseModel):
    email: str = Field(..., description="The email of the user to associate the chat with.")
    messages: List[ChatMessage] = Field(..., description="The list of messages in the conversation.")


# --- LangChain Chain Definition ---
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Respond to the user's query concisely."),
    ("human", "{input}"),
])
chain = prompt_template | chat_model


# --- API Endpoint Definitions ---
@app.post("/invoke", response_model=LLMResponse, summary="Invoke the LLM with a prompt")
async def invoke_llm(request: PromptRequest):
    """Handles real-time chat messages by sending them to the Groq LLM."""
    try:
        llm_result = chain.invoke({"input": request.text})
        if not llm_result.content:
             raise HTTPException(status_code=500, detail="LLM returned an empty response.")
        return {"response": llm_result.content}
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save-chat", summary="Save chat history to Cloudflare R2")
async def save_chat(request: ChatHistoryRequest):
    """
    Saves the provided chat history as a JSON file in a user-specific folder
    within the Cloudflare R2 bucket.
    """
    try:
        # Generate a unique filename using a UTC timestamp (e.g., 2025-10-02T12-30-00.json)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        
        # Create a structured file path (object key) in R2, organized by email.
        object_key = f"chats/{request.email}/{timestamp}.json"
        
        # Convert the message objects into a standard list of dictionaries
        chat_data = [msg.model_dump() for msg in request.messages]
        
        # Serialize the chat data to a JSON formatted string
        chat_body = json.dumps(chat_data, indent=2)

        # Use the boto3 S3 client to upload the JSON string as a file to R2
        s3_client.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=object_key,
            Body=chat_body,
            ContentType='application/json'
        )
        
        print(f"✅ Successfully saved chat to R2: {object_key}")
        return {"status": "success", "message": "Chat history saved successfully.", "key": object_key}

    except Exception as e:
        print(f"❌ Error saving chat to R2: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while saving chat history: {e}")


# --- Main entry point to run the server ---
if __name__ == "__main__":
    print("🚀 Starting FastAPI server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)

