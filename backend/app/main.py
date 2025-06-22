from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from typing import List, Dict, Any, Optional
from .services import rag_service
from fastapi.responses import StreamingResponse
import time
import asyncio
import threading

app = FastAPI()

# CORS configuration
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to store uploaded PDFs
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Cache for recently retrieved context blocks (query -> blocks)
context_blocks_cache: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = 300  # 5 minutes TTL for cache entries

# Global variable to store pre-loaded QA blocks
preloaded_qa_blocks = []
is_system_warmed_up = False

class Query(BaseModel):
    query: str

@app.on_event("startup")
async def startup_event():
    """Pre-warm the system by loading models and parsing PDFs in a background thread"""
    def warmup_system():
        global preloaded_qa_blocks, is_system_warmed_up
        print("🔥 Starting system warm-up...")
        
        # Initialize embedding model to load it into memory
        print("📊 Initializing embedding model...")
        rag_service.initialize_embedding_model()
        
        # Pre-load PDFs and parse them into QA blocks
        print("📑 Pre-loading PDFs...")
        pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
        
        if pdf_files:
            all_blocks = []
            for filename in pdf_files:
                print(f"⏳ Processing {filename}...")
                file_path = os.path.join(DATA_DIR, filename)
                try:
                    blocks = rag_service.parse_pdf_into_qa_blocks(file_path, filename)
                    all_blocks.extend(blocks)
                    print(f"✅ Successfully processed {filename}, found {len(blocks)} QA blocks")
                except Exception as e:
                    print(f"❌ Error processing {filename}: {str(e)}")
            
            preloaded_qa_blocks = all_blocks
            print(f"🎉 System warm-up complete! Pre-loaded {len(preloaded_qa_blocks)} QA blocks from {len(pdf_files)} PDFs")
        else:
            print("ℹ️ No PDFs found in the data directory. System warm-up limited to model initialization.")
        
        is_system_warmed_up = True
    
    # Run the warm-up in a background thread to avoid blocking the server startup
    thread = threading.Thread(target=warmup_system)
    thread.daemon = True
    thread.start()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Agentic RAG API"}

@app.get("/documents", response_model=List[str])
def get_documents():
    """Returns a list of PDF documents in the knowledge base."""
    return [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]

@app.get("/system/status")
def get_system_status():
    """Returns the current system status"""
    return {
        "is_warmed_up": is_system_warmed_up,
        "preloaded_qa_blocks_count": len(preloaded_qa_blocks),
        "cache_entries_count": len(context_blocks_cache),
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global preloaded_qa_blocks
    
    file_path = os.path.join(DATA_DIR, file.filename)
    if os.path.exists(file_path):
        raise HTTPException(status_code=400, detail=f"File '{file.filename}' already exists.")
        
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the new file and add its blocks to preloaded_qa_blocks
    try:
        new_blocks = rag_service.parse_pdf_into_qa_blocks(file_path, file.filename)
        preloaded_qa_blocks.extend(new_blocks)
        print(f"Added {len(new_blocks)} QA blocks from newly uploaded file {file.filename}")
    except Exception as e:
        print(f"Error processing newly uploaded file {file.filename}: {str(e)}")
        
    return {"filename": file.filename, "content_type": file.content_type}

@app.delete("/documents/{filename}")
def delete_document(filename: str):
    global preloaded_qa_blocks
    
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    
    os.remove(file_path)
    
    # Remove QA blocks for the deleted file from preloaded_qa_blocks
    preloaded_qa_blocks = [block for block in preloaded_qa_blocks if block.source_document != filename]
    print(f"Removed QA blocks for deleted file {filename}. Remaining blocks: {len(preloaded_qa_blocks)}")
    
    # Clear the context blocks cache as it may contain blocks from the deleted file
    context_blocks_cache.clear()
    
    return {"message": f"Successfully deleted '{filename}'"}

def get_context_blocks_for_query(query: str):
    """Helper function to get context blocks for a query, using cache when possible."""
    # Check if we have a recent cache entry for this query
    if query in context_blocks_cache:
        cache_entry = context_blocks_cache[query]
        if time.time() - cache_entry["timestamp"] < CACHE_TTL:
            print(f"Using cached context blocks for query: {query[:50]}...")
            return cache_entry["all_qa_blocks"], cache_entry["best_matches"]
    
    # If system is warmed up, use preloaded QA blocks
    if is_system_warmed_up and preloaded_qa_blocks:
        print(f"Using preloaded QA blocks for query: {query[:50]}...")
        all_qa_blocks = preloaded_qa_blocks
    else:
        # Otherwise, load from files (this should rarely happen if system is warmed up)
        print(f"System not warmed up yet, loading QA blocks from files for query: {query[:50]}...")
        pdf_files = get_documents()
        if not pdf_files:
            return None, None
            
        all_qa_blocks = []
        for filename in pdf_files:
            file_path = os.path.join(DATA_DIR, filename)
            all_qa_blocks.extend(rag_service.parse_pdf_into_qa_blocks(file_path, filename))
    
    if not all_qa_blocks:
        return None, None
        
    best_matches = rag_service.find_best_matches(query, all_qa_blocks, top_n=3)
    
    # Store in cache
    context_blocks_cache[query] = {
        "all_qa_blocks": all_qa_blocks,
        "best_matches": best_matches,
        "timestamp": time.time()
    }
    
    return all_qa_blocks, best_matches

@app.post("/ask/stream")
async def ask_stream(query: Query):
    """Stream the LLM response, without images or sources."""
    all_qa_blocks, best_matches = get_context_blocks_for_query(query.query)
    
    if not all_qa_blocks:
        def error_stream():
            yield "Could not extract any processable Q&A content from the documents in the knowledge base."
        return StreamingResponse(error_stream())

    if not best_matches:
        def no_matches_stream():
            yield "I couldn't find a relevant answer in the knowledge base for your question."
        return StreamingResponse(no_matches_stream())

    # Get streaming response from the LLM
    return StreamingResponse(
        rag_service.get_streaming_answer_from_llm(query.query, best_matches),
        media_type="text/plain"
    )

@app.post("/ask")
def ask(query: Query):
    all_qa_blocks, best_matches = get_context_blocks_for_query(query.query)
    
    if not all_qa_blocks:
        return {"llm_answer": "Could not extract any processable Q&A content from the documents in the knowledge base.", "sources": []}

    if not best_matches:
        return {"llm_answer": "I couldn't find a relevant answer in the knowledge base for your question.", "sources": []}

    # Get a synthesized answer from the LLM based on the best blocks
    agentic_response = rag_service.get_agentic_answer_from_llm(query.query, best_matches)
    
    # Include the original retrieved content for context on the frontend
    retrieved_context = [block.to_dict() for block in best_matches]
    agentic_response["retrieved_context"] = retrieved_context
    
    # Add question-specific images to each source in the LLM response
    if "sources" in agentic_response:
        for source in agentic_response["sources"]:
            # Extract question number from the LLM's source
            question_text = source.get("question", "")
            import re
            match = re.match(r'^(Q\d+)', question_text, re.IGNORECASE)
            if match:
                question_number = match.group(1).upper()
                print(f"Looking for images for {question_number} from LLM source")
                # Find images for this specific question number
                question_images = rag_service.find_images_for_question_number(question_number, all_qa_blocks)
                source["images"] = question_images
                print(f"Added {len(question_images)} images for {question_number}")
            else:
                source["images"] = []

    return agentic_response 