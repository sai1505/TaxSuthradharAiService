import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# LangChain Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# --- NEW IMPORTS FOR RAG CHAIN ---
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Standard Library Imports
import os
from dotenv import load_dotenv
from typing import List, Any, Optional
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import tempfile
from langchain.schema import Document as LangChainDocument
from docling.document_converter import DocumentConverter
import docling

import zipfile
import io
import shutil
import traceback

# Import the configured S3 client and bucket name from our config file
from r2_config import s3_client, R2_BUCKET_NAME

# This dictionary holds the processed document retriever for each user.
user_retrievers = {}

# --- FastAPI App Initialization ---
app = FastAPI(
    title="PythonChatMicroservice",
    description="An API that uses Groq for chat and saves conversations to Cloudflare R2.",
    version="1.3.0", # Version bump to reflect new RAG feature
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
        model_name="llama-3.1-8b-instant",
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize ChatGroq model: {e}")

# --- Pydantic Models for Data Validation (No Changes) ---
class PromptRequest(BaseModel):
    text: str = Field(..., min_length=1)

class LLMResponse(BaseModel):
    response: str

class ChatMessage(BaseModel):
    id: Any
    text: str
    sender: str

class ChatHistoryRequest(BaseModel):
    email: str = Field(...)
    messages: List[ChatMessage] = Field(...)
    chat_id: Optional[str] = Field(None)
    # --- NEW FIELD ---
    index_key: Optional[str] = Field(None, description="The R2 key of the associated FAISS index zip file.")

class ChatMetadata(BaseModel):
    id: str
    title: str
    date: str

class ChatHistoryResponse(BaseModel):
    messages: List[ChatMessage]

class LoadContextRequest(BaseModel):
    chat_id: str # This is the R2 key for the chat JSON file
    email: str

# --- LangChain Chain Definition ---
# This is the chain for general conversation when NO document is uploaded.
general_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Respond to the user's query concisely."),
    ("human", "{input}"),
])
general_chain = general_prompt | chat_model

# --- NEW: This is the prompt template for answering from a document. ---
rag_prompt = ChatPromptTemplate.from_template("""
Answer the user's question based ONLY on the following context:

<context>
{context}
</context>

Question: {input}
""")

# --- API Endpoint Definitions ---

# --- MODIFIED: The /invoke endpoint now requires an email and handles RAG ---
@app.post("/invoke/{email}", response_model=LLMResponse, summary="Invoke LLM with optional document context")
async def invoke_llm(email: str, request: PromptRequest):
    """
    Handles chat messages. If a document has been uploaded for the user,
    it uses the RAG chain. Otherwise, it uses the general conversation chain.
    """
    try:
        response_content = ""
        # Check if a retriever exists for this user
        if email in user_retrievers:
            print(f"🧠 Using RAG chain for user: {email}")
            retriever = user_retrievers[email]
            
            # Create the chains specifically for this RAG query
            document_chain = create_stuff_documents_chain(chat_model, rag_prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            # Invoke the RAG chain
            llm_result = await retrieval_chain.ainvoke({"input": request.text})
            # The RAG chain returns the answer in a dictionary with the key 'answer'
            response_content = llm_result.get('answer', "I couldn't find an answer in the document.")

        else:
            print(f"💬 Using general chain for user: {email}")
            # Invoke the general chain for a normal conversation
            llm_result = await general_chain.ainvoke({"input": request.text})
            response_content = llm_result.content

        if not response_content:
             raise HTTPException(status_code=500, detail="LLM returned an empty response.")
        return {"response": response_content}

    except Exception as e:
        print(f"Error during LLM invocation for {email}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- /save-chat endpoint (No Changes) ---
@app.post("/save-chat", summary="Save or update chat history in Cloudflare R2")
async def save_chat(request: ChatHistoryRequest):
    try:
        object_key = ""
        if request.chat_id:
            if not request.chat_id.startswith(f"chats/{request.email}/"):
                 raise HTTPException(status_code=403, detail="Access denied: Invalid chat_id for the user.")
            object_key = request.chat_id
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
            object_key = f"chats/{request.email}/{timestamp}.json"

        # Create a single dictionary containing the index_key and messages
        chat_content = {
            "index_key": request.index_key,
            "messages": [msg.model_dump() for msg in request.messages]
        }
        chat_body = json.dumps(chat_content, indent=2)
        
        s3_client.put_object(
            Bucket=R2_BUCKET_NAME, Key=object_key, Body=chat_body, ContentType='application/json'
        )
        
        print(f"✅ Saved chat with index_key '{request.index_key}' to R2: {object_key}")
        return {"status": "success", "key": object_key}

    except Exception as e:
        print(f"❌ Error saving chat to R2: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while saving chat history: {e}")

# --- /upload-document endpoint (No Changes) ---
@app.post("/upload-document/{email}", summary="Upload a PDF and persist its vector index")
async def upload_document(email: str, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Initialize variables to None for robust cleanup
    tmp_file_path = None
    temp_dir = None
    temp_zip_path = None
    
    try:
        # 1. Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name
        
        # 2. Process the document with Docling
        print(f"📄 Processing file with Docling: {tmp_file_path}")
        converter = DocumentConverter()
        result = converter.convert(tmp_file_path)
        text_content = result.document.export_to_markdown()
        docs = [LangChainDocument(page_content=text_content)]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

        # 3. Save the FAISS index to a temporary local directory
        temp_dir = tempfile.mkdtemp()
        vectorstore.save_local(temp_dir)
        print(f"✅ FAISS index saved locally to {temp_dir}")

        # 4. Zip the directory
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        # Create a temporary path for the zip file that doesn't conflict with other runs
        zip_filename_base = os.path.join(tempfile.gettempdir(), f"index-{timestamp}")
        temp_zip_path = shutil.make_archive(zip_filename_base, 'zip', temp_dir)

        # 5. Upload the zip file to R2
        index_key = f"indexes/{email}/{os.path.basename(temp_zip_path)}"
        with open(temp_zip_path, 'rb') as zip_file:
            s3_client.put_object(
                Bucket=R2_BUCKET_NAME, Key=index_key, Body=zip_file, ContentType='application/zip'
            )
        print(f"✅ Zipped index uploaded to R2 with key: {index_key}")

        # 6. Load retriever into memory for immediate use & return key
        user_retrievers[email] = vectorstore.as_retriever()
        
        return {
            "status": "success",
            "message": f"File '{file.filename}' processed successfully.",
            "index_key": index_key
        }
    # --- THIS IS THE CORRECTED PART ---
    except Exception as e:
        # Log the full error to your server console for debugging
        print(f"❌ Error processing document for {email}: {e}")
        traceback.print_exc()
        
        # Raise an HTTP exception to inform the frontend that something went wrong
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")
    finally:
        # This cleanup logic is now safer
        print("--- Running cleanup ---")
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
            print(f"Cleaned up temp file: {tmp_file_path}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temp directory: {temp_dir}")
        if temp_zip_path and os.path.exists(temp_zip_path):
            os.unlink(temp_zip_path)
            print(f"Cleaned up temp zip: {temp_zip_path}")

@app.post("/load-context", summary="Load a persisted document index into memory")
async def load_context(request: LoadContextRequest):
    temp_dir = None
    try:
        chat_obj = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=request.chat_id)
        chat_content = json.loads(chat_obj['Body'].read().decode('utf-8'))
        index_key = chat_content.get("index_key")

        if not index_key:
            if request.email in user_retrievers:
                del user_retrievers[request.email]
            return {"status": "success", "message": "No document context to load for this chat."}

        index_obj = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=index_key)
        
        temp_dir = tempfile.mkdtemp()
        with io.BytesIO(index_obj['Body'].read()) as zip_buffer:
            with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
        
        user_retrievers[request.email] = vectorstore.as_retriever()
        
        print(f"✅ Successfully loaded index '{index_key}' for user '{request.email}'")
        return {"status": "success", "message": "Document context loaded."}
    except Exception as e:
        # Proper error logging and response
        print(f"❌ Error loading context for {request.email}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to load document context.")
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


# --- /chats/{email} endpoint (No Changes) ---
@app.get("/chats/{email}", response_model=List[ChatMetadata], summary="List all chat histories for a user")
async def get_chat_list(email: str):
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
                # Load the entire chat object
                chat_content = json.loads(file_obj['Body'].read().decode('utf-8'))
                # Access the 'messages' key to find the first message
                messages = chat_content.get("messages", [])
                
                first_message = next((msg['text'] for msg in messages if msg['sender'] == 'user'), "Untitled Chat")
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

# --- /chat endpoint (No Changes) ---
@app.get("/chat", response_model=dict, summary="Retrieve a specific chat conversation")
async def get_chat_content(key: str):
    if not key.startswith("chats/") or ".." in key:
        raise HTTPException(status_code=400, detail="Invalid chat key specified.")
    try:
        file_obj = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=key)
        chat_content = json.loads(file_obj['Body'].read().decode('utf-8'))
        # Return the entire object, not just messages
        return chat_content
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Chat history not found.")
    except Exception as e:
        print(f"❌ Error retrieving chat {key}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat content: {e}")

# --- Main entry point (No Changes) ---
if __name__ == "__main__":
    print("🚀 Starting FastAPI server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)