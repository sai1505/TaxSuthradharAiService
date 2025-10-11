import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from typing import List, Any, Optional
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

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
    allow_origins=["*"], # In production, restrict this to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Groq Model Initialization ---
# (Your existing Groq setup code remains here)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set!")

try:
    chat_model = ChatGroq(
        temperature=0.7,
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
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

class ChatHistoryRequest(BaseModel):
    email: str = Field(..., description="The email of the user to associate the chat with.")
    messages: List[ChatMessage] = Field(..., description="The list of messages in the conversation.")
    chat_id: Optional[str] = Field(None, description="The existing chat ID (R2 object key) to overwrite. If null, a new chat is created.")

# --- NEW: Models for retrieving history ---
class ChatMetadata(BaseModel):
    id: str  # This will be the R2 object key (e.g., chats/user@email.com/timestamp.json)
    title: str
    date: str

class ChatHistoryResponse(BaseModel):
    messages: List[ChatMessage]

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

@app.post("/save-chat", summary="Save or update chat history in Cloudflare R2")
async def save_chat(request: ChatHistoryRequest):
    """
    Saves the chat history. If a chat_id is provided, it overwrites the existing
    chat file. Otherwise, it creates a new one.
    """
    try:
        # Determine the object key: use existing or create a new one
        if request.chat_id:
            # Security check: ensure the provided key belongs to the user
            if not request.chat_id.startswith(f"chats/{request.email}/"):
                 raise HTTPException(status_code=403, detail="Access denied: Invalid chat_id for the user.")
            object_key = request.chat_id
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
            object_key = f"chats/{request.email}/{timestamp}.json"

        chat_data = [msg.model_dump() for msg in request.messages]
        chat_body = json.dumps(chat_data, indent=2)
        
        s3_client.put_object(
            Bucket=R2_BUCKET_NAME, Key=object_key, Body=chat_body, ContentType='application/json'
        )
        
        print(f"✅ Successfully saved/updated chat in R2: {object_key}")
        # ALWAYS return the key so the frontend can track it
        return {"status": "success", "message": "Chat history saved successfully.", "key": object_key}
    except Exception as e:
        print(f"❌ Error saving chat to R2: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while saving chat history: {e}")


# --- NEW ENDPOINT: Get list of chats for a user ---
@app.get("/chats/{email}", response_model=List[ChatMetadata], summary="List all chat histories for a user")
async def get_chat_list(email: str):
    """
    Retrieves a list of all chat sessions for a given user email from R2.
    It generates a title from the first message of each chat.
    """
    chat_list = []
    prefix = f"chats/{email}/"
    try:
        response = s3_client.list_objects_v2(Bucket=R2_BUCKET_NAME, Prefix=prefix)
        if 'Contents' not in response:
            return []
        
        sorted_objects = sorted(response['Contents'], key=lambda obj: obj['LastModified'], reverse=True)
        ist_tz = ZoneInfo("Asia/Kolkata")

        for obj in sorted_objects:
            object_key = obj['Key']
            try:
                file_obj = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=object_key)
                chat_content = json.loads(file_obj['Body'].read().decode('utf-8'))
                
                first_message = next((msg['text'] for msg in chat_content if msg['sender'] == 'user'), "Untitled Chat")
                title = (first_message[:50] + '...') if len(first_message) > 50 else first_message
                local_time = obj['LastModified'].astimezone(ist_tz)
                date_str = local_time.strftime("%B %d, %Y, %I:%M %p")

                chat_list.append(ChatMetadata(id=object_key, title=title, date=date_str))
            except Exception as e:
                print(f"Skipping file {object_key} due to parsing error: {e}")
                continue
                
        return chat_list
    except Exception as e:
        print(f"❌ Error listing chats for {email}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat list: {e}")

# --- NEW ENDPOINT: Get content of a specific chat ---
@app.get("/chat", response_model=ChatHistoryResponse, summary="Retrieve a specific chat conversation")
async def get_chat_content(key: str):
    """
    Retrieves the full message history for a specific chat, identified by its R2 object key.
    """
    if not key.startswith("chats/") or ".." in key:
        raise HTTPException(status_code=400, detail="Invalid chat key specified.")
    try:
        file_obj = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=key)
        chat_content = json.loads(file_obj['Body'].read().decode('utf-8'))
        return ChatHistoryResponse(messages=chat_content)
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Chat history not found.")
    except Exception as e:
        print(f"❌ Error retrieving chat {key}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat content: {e}")


# --- Main entry point to run the server ---
if __name__ == "__main__":
    print("🚀 Starting FastAPI server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)