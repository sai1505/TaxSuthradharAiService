# main.py - Configured for Docling and Tavily AI

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# LangChain Imports
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain.schema import Document as LangChainDocument

# Standard Library Imports
import os
from dotenv import load_dotenv
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import tempfile
import zipfile
import io
import shutil
import traceback
from typing import List, Any, Optional, Dict

# --- PDF Processing Library ---
from docling.document_converter import DocumentConverter
import docling

# Import the configured S3 client and bucket name
from r2_config import s3_client, R2_BUCKET_NAME

# --- Load Environment Variables ---
load_dotenv()

# This dictionary holds the processed document retriever for each user.
user_retrievers = {}

# --- Initialize Tools ---
web_retriever = TavilySearchAPIRetriever(k=3)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="PythonChatMicroservice",
    description="An API that uses Groq, RAG (with Docling), and Tavily web search.",
    version="1.7.0",
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Groq Model & API Key Initialization ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set!")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY environment variable not set!")

try:
    chat_model = ChatGroq(
        temperature=0.7,
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize ChatGroq model: {e}")

# --- Pydantic Models (No Changes) ---
class PromptRequest(BaseModel):
    text: str = Field(..., min_length=1)
class LLMResponse(BaseModel):
    response: str
class ChatMessage(BaseModel):
    id: Any; text: str; sender: str
class ChatHistoryRequest(BaseModel):
    email: str = Field(...); messages: List[ChatMessage] = Field(...)
    chat_id: Optional[str] = Field(None); index_key: Optional[str] = Field(None)
class ChatMetadata(BaseModel):
    id: str; title: str; date: str
class LoadContextRequest(BaseModel):
    chat_id: str; email: str


# --- ROUTING AND ADVANCED CHAIN DEFINITIONS ---

# 1. Chain for Document-Specific Questions
rag_prompt = ChatPromptTemplate.from_template("""
You are an expert AI assistant. Answer the user's question based ONLY on the context from their uploaded document provided below.

<context>
{context}
</context>

Question: {input}
""")
document_chain = create_stuff_documents_chain(chat_model, rag_prompt)

# 2. Chain for Web Search Questions
web_search_prompt = ChatPromptTemplate.from_template("""
You are an expert Indian tax consultant specializing in helping IT employees
legally save tax under the Income Tax Act.

### Context:
{context}

Based on the context and retrieved knowledge:
1. Identify all applicable legal tax-saving options (e.g., 80C, 80D, 80CCD, 24B, HRA, etc.).
2. Suggest compliant investment and deduction strategies.
3. Indicate whether the Old or New tax regime is better for this employee.
4. Provide a brief legal explanation for each recommendation.
5. End with a simple step-by-step action checklist for saving tax legally.

Ensure your advice:
- Is strictly within Indian tax laws.
- Avoids any unethical or illegal suggestions.
- Is concise, accurate, and actionable.

Question: {input}
""")
web_search_chain = create_stuff_documents_chain(chat_model, web_search_prompt)

# 3. Router to decide which chain to use
router_prompt = PromptTemplate.from_template(
    'Given the user\'s question, classify it as "document_specific" or "web_search".\n'
    '- If about personal details or data in their uploaded document, classify as "document_specific".\n'
    '- If about general tax laws, saving strategies, or public info, classify as "web_search".\n'
    "Question: {input}\nClassification:"
)
router_chain = router_prompt | chat_model

# --- /invoke endpoint using Tavily Router ---
@app.post("/invoke/{email}", response_model=LLMResponse, summary="Invoke LLM with intelligent routing")
async def invoke_llm(email: str, request: PromptRequest):
    try:
        retriever = user_retrievers.get(email)

        if not retriever:
            print(f"💬 No document for {email}. Using Tavily web search.")
            retrieval_chain = create_retrieval_chain(web_retriever, web_search_chain)
            llm_result = await retrieval_chain.ainvoke({"input": request.text})
            return LLMResponse(response=llm_result.get('answer', "Sorry, I couldn't find an answer."))

        print(f"🤔 Routing question for user: {email}")
        classification_result = await router_chain.ainvoke({"input": request.text})
        classification = classification_result.content.strip().lower()

        if "document_specific" in classification:
            print("🧠 Router chose: Document. Answering from user's file.")
            final_chain = create_retrieval_chain(retriever, document_chain)
        else:
            print("🌐 Router chose: Web Search (Tavily). Answering with live data.")
            final_chain = create_retrieval_chain(web_retriever, web_search_chain)

        llm_result = await final_chain.ainvoke({"input": request.text})
        response_content = llm_result.get('answer', "I couldn't find a reliable answer.")
        
        if not response_content:
             raise HTTPException(status_code=500, detail="LLM returned an empty response.")
        return LLMResponse(response=response_content)

    except Exception as e:
        print(f"Error during LLM invocation for {email}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --- /upload-document endpoint using Docling ---
@app.post("/upload-document/{email}", summary="Upload a PDF and persist its vector index")
async def upload_document(email: str, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    tmp_file_path, temp_dir, temp_zip_path = None, None, None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name

        converter = DocumentConverter()
        result = converter.convert(tmp_file_path)
        text_content = result.document.export_to_markdown()
        
        docs = [LangChainDocument(page_content=text_content)]
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

        temp_dir = tempfile.mkdtemp()
        vectorstore.save_local(temp_dir)

        # --- NEW FILENAME LOGIC ---
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        original_filename_base = os.path.splitext(file.filename)[0].replace(" ", "_")
        zip_filename = f"{original_filename_base}-{timestamp}"
        
        temp_zip_base_path = os.path.join(tempfile.gettempdir(), zip_filename)
        temp_zip_path = shutil.make_archive(temp_zip_base_path, 'zip', temp_dir)

        index_key = f"indexes/{email}/{os.path.basename(temp_zip_path)}"
        with open(temp_zip_path, 'rb') as zip_file:
            s3_client.put_object(Bucket=R2_BUCKET_NAME, Key=index_key, Body=zip_file, ContentType='application/zip')
        print(f"✅ Zipped index uploaded to R2 with key: {index_key}")

        user_retrievers[email] = vectorstore.as_retriever()
        return {"status": "success", "message": f"File '{file.filename}' processed.", "index_key": index_key}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path): os.unlink(tmp_file_path)
        if temp_dir and os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        if temp_zip_path and os.path.exists(temp_zip_path): os.unlink(temp_zip_path)

# --- Other endpoints remain unchanged ---
# (The code for save_chat, load_context, get_chat_list, and get_chat_content is identical)
@app.post("/save-chat", summary="Save or update chat history in Cloudflare R2")
async def save_chat(request: ChatHistoryRequest):
    try:
        object_key = request.chat_id or f"chats/{request.email}/{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')}.json"
        if request.chat_id and not request.chat_id.startswith(f"chats/{request.email}/"):
            raise HTTPException(status_code=403, detail="Access denied: Invalid chat_id for the user.")
        
        chat_content = {"index_key": request.index_key, "messages": [msg.model_dump() for msg in request.messages]}
        s3_client.put_object(Bucket=R2_BUCKET_NAME, Key=object_key, Body=json.dumps(chat_content, indent=2), ContentType='application/json')
        print(f"✅ Saved chat to R2: {object_key}")
        return {"status": "success", "key": object_key}
    except Exception as e:
        print(f"❌ Error saving chat to R2: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-context", summary="Load a persisted document index into memory")
async def load_context(request: LoadContextRequest):
    temp_dir = None
    try:
        chat_obj = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=request.chat_id)
        chat_content = json.loads(chat_obj['Body'].read().decode('utf-8'))
        index_key = chat_content.get("index_key")

        if not index_key:
            if request.email in user_retrievers: del user_retrievers[request.email]
            return {"status": "success", "message": "No document context to load for this chat."}

        index_obj = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=index_key)
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(io.BytesIO(index_obj['Body'].read()), 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(temp_dir, embeddings, allow_dangerous_deserialization=True)
        user_retrievers[request.email] = vectorstore.as_retriever()
        print(f"✅ Successfully loaded index '{index_key}' for user '{request.email}'")
        return {"status": "success", "message": "Document context loaded."}
    except Exception as e:
        print(f"❌ Error loading context for {request.email}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to load document context.")
    finally:
        if temp_dir and os.path.exists(temp_dir): shutil.rmtree(temp_dir)

@app.get("/chats/{email}", response_model=List[ChatMetadata], summary="List all chat histories for a user")
async def get_chat_list(email: str):
    try:
        prefix = f"chats/{email}/"
        response = s3_client.list_objects_v2(Bucket=R2_BUCKET_NAME, Prefix=prefix)
        if 'Contents' not in response: return []
        
        sorted_objects = sorted(response['Contents'], key=lambda obj: obj['LastModified'], reverse=True)
        ist_tz = ZoneInfo("Asia/Kolkata")
        chat_list = []

        for obj in sorted_objects:
            try:
                file_obj = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=obj['Key'])
                messages = json.loads(file_obj['Body'].read().decode('utf-8')).get("messages", [])
                first_message = next((msg['text'] for msg in messages if msg['sender'] == 'user'), "Untitled Chat")
                title = (first_message[:50] + '...') if len(first_message) > 50 else first_message
                date_str = obj['LastModified'].astimezone(ist_tz).strftime("%B %d, %Y, %I:%M %p")
                chat_list.append(ChatMetadata(id=obj['Key'], title=title, date=date_str))
            except Exception as e:
                print(f"Skipping file {obj['Key']} due to parsing error: {e}")
        return chat_list
    except Exception as e:
        print(f"❌ Error listing chats for {email}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat list: {e}")

@app.get("/chat", response_model=dict, summary="Retrieve a specific chat conversation")
async def get_chat_content(key: str):
    if not key.startswith("chats/") or ".." in key:
        raise HTTPException(status_code=400, detail="Invalid chat key specified.")
    try:
        file_obj = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=key)
        return json.loads(file_obj['Body'].read().decode('utf-8'))
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Chat history not found.")
    except Exception as e:
        print(f"❌ Error retrieving chat {key}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat content: {e}")

@app.delete("/delete-chat/{email}", summary="Delete a specific chat history and its document index")
async def delete_chat(email: str, key: str):
    if not key.startswith(f"chats/{email}/"):
        raise HTTPException(status_code=403, detail="Access denied.")
    
    try:
        # 1. Get the chat object to find the associated index_key
        chat_obj = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=key)
        chat_content = json.loads(chat_obj['Body'].read().decode('utf-8'))
        index_key_to_delete = chat_content.get("index_key")

        # 2. If an index exists, delete it
        if index_key_to_delete:
            s3_client.delete_object(Bucket=R2_BUCKET_NAME, Key=index_key_to_delete)
            print(f"✅ Deleted associated index: {index_key_to_delete}")
        
        # 3. Delete the main chat JSON file
        s3_client.delete_object(Bucket=R2_BUCKET_NAME, Key=key)
        print(f"✅ Deleted chat object: {key}")
        
        return {"status": "success", "message": f"Deleted chat {key}"}
    except s3_client.exceptions.NoSuchKey:
        return {"status": "success", "message": f"Chat {key} not found, presumed deleted."}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to delete chat: {e}")

# --- NEW: Endpoint to delete all chats for a user ---
@app.delete("/chats/clear/{email}", summary="Delete all chat histories and document indexes for a user")
async def clear_all_chats(email: str):
    try:
        # Batch delete all chat files
        chat_prefix = f"chats/{email}/"
        chat_objects = s3_client.list_objects_v2(Bucket=R2_BUCKET_NAME, Prefix=chat_prefix).get('Contents', [])
        if chat_objects:
            chats_to_delete = [{'Key': obj['Key']} for obj in chat_objects]
            s3_client.delete_objects(Bucket=R2_BUCKET_NAME, Delete={'Objects': chats_to_delete})
            print(f"✅ Cleared {len(chats_to_delete)} chats for user {email}")

        # Batch delete all index files
        index_prefix = f"indexes/{email}/"
        index_objects = s3_client.list_objects_v2(Bucket=R2_BUCKET_NAME, Prefix=index_prefix).get('Contents', [])
        if index_objects:
            indexes_to_delete = [{'Key': obj['Key']} for obj in index_objects]
            s3_client.delete_objects(Bucket=R2_BUCKET_NAME, Delete={'Objects': indexes_to_delete})
            print(f"✅ Cleared {len(indexes_to_delete)} document indexes for user {email}")

        return {"status": "success", "message": "All user data cleared."}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to clear user data: {e}")

@app.get("/documents/{email}", summary="List all uploaded document indexes for a user")
async def get_document_list(email: str):
    """
    Lists all document index files for a user, parsing the original filename
    and providing the upload date.
    """
    document_list = []
    prefix = f"indexes/{email}/"
    try:
        response = s3_client.list_objects_v2(Bucket=R2_BUCKET_NAME, Prefix=prefix)
        if 'Contents' not in response:
            return []
        
        # Sort by most recently uploaded
        sorted_objects = sorted(response['Contents'], key=lambda obj: obj['LastModified'], reverse=True)
        ist_tz = ZoneInfo("Asia/Kolkata")

        for obj in sorted_objects:
            object_key = obj['Key']
            # --- Filename Parsing Logic ---
            # Extracts the original filename from the R2 key
            # e.g., "indexes/user@test.com/My_ITR_Form-20251012-103000.zip" -> "My_ITR_Form.pdf"
            filename_with_ext = os.path.basename(object_key)
            # Find the last hyphen, which separates the name from the timestamp
            last_hyphen_index = filename_with_ext.rfind('-')
            if last_hyphen_index != -1:
                original_filename_base = filename_with_ext[:last_hyphen_index]
                # Replace underscores back with spaces for display and assume it was a PDF
                display_filename = f"{original_filename_base.replace('_', ' ')}.pdf"
            else:
                # Fallback in case the format is unexpected
                display_filename = filename_with_ext

            local_time = obj['LastModified'].astimezone(ist_tz)
            date_str = local_time.strftime("%B %d, %Y") # Format date nicely

            document_list.append({
                "id": object_key,
                "filename": display_filename,
                "uploadDate": date_str
            })
            
        return document_list
    except Exception as e:
        print(f"❌ Error listing documents for {email}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to retrieve document list.")

@app.delete("/document/{email}", summary="Delete a specific document index")
async def delete_document(email: str, key: str):
    """
    Deletes a single document index file from R2, ensuring the user owns it.
    The object key is passed as a query parameter.
    """
    # Security check: Ensure the key belongs to the user trying to delete it.
    if not key.startswith(f"indexes/{email}/"):
        raise HTTPException(status_code=403, detail="Access denied: You do not have permission to delete this document.")
    
    try:
        s3_client.delete_object(Bucket=R2_BUCKET_NAME, Key=key)
        print(f"✅ Deleted document index: {key}")
        return {"status": "success", "message": f"Deleted document {key}"}
    except Exception as e:
        print(f"❌ Error deleting document {key}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}")

# --- Main entry point ---
if __name__ == "__main__":
    print("🚀 Starting FastAPI server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)