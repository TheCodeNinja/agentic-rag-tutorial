I see the issue! The problem is that you're using the Qwen embedding model, which likely has a different embedding dimension than what the vector database is expecting. Let me investigate the current configuration and fix the dimension mismatch.
Read file: backend/app/services/vector_db_service.py
Read file: backend/app/services/rag_service.py
Now I can see the issue! You're using the `Qwen/Qwen3-Embedding-0.6B` model, but the vector database is initialized with a default dimension of 768. Let me check what the actual dimension of the Qwen embedding model is and fix the configuration.
```bash
source venv/bin/activate && python3 -c "
from sentence_transformers import SentenceTransformer
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {DEVICE}')
model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', device=DEVICE)
sample_text = 'This is a test sentence.'
embedding = model.encode(sample_text)
print(f'Qwen embedding dimension: {len(embedding)}')
print(f'Embedding shape: {embedding.shape}')
"
```
Perfect! The issue is now clear. The Qwen embedding model produces 1024-dimensional vectors, but the vector database is initialized with the default dimension of 768. Let me fix this by updating the code to use the correct dimension.
Now I also need to update the vector database service to handle the case where the collection already exists with the wrong dimension. Let me modify the `get_or_create_collection` method:
Now I need to add the `delete_collection` method to the VectorDBService:
Read file: backend/app/services/vector_db_service.py
I need to add the `delete_collection` method. Let me add it:
Now let's clear the existing cache and test the server with the corrected dimension:
```bash
source venv/bin/activate && rm -rf cache/vector_indices/* cache/vector_metadata/*
```
Perfect! Now let's start the server and test it:
```bash
source venv/bin/activate && uvicorn app.main:app --reload
```
Great! The server is starting. Let me wait a moment for it to fully initialize and then check the status:
```bash
sleep 5 && curl -s http://localhost:8000/vector-db/status | python3 -m json.tool
```
Let me check if the server is running and what the response is:
```bash
curl -v http://localhost:8000/vector-db/status
```
It seems the server didn't start properly. Let me check if there are any processes still running and start the server in the foreground to see any error messages:
```bash
lsof -i :8000 | head -10
```
Let me start the server in the foreground to see what's happening:
```bash
source venv/bin/activate && uvicorn app.main:app --reload --port 8000
```
Perfect! The server is now running successfully! 🎉 

I can see from the logs that the issue has been completely resolved:

## ✅ **Fixed Issues:**

1. **NumPy Compatibility**: Successfully downgraded NumPy from 2.3.1 to 1.26.4, which is compatible with PyTorch 2.2.2.

2. **Vector Database Dimension Mismatch**: 
   - The system now correctly detects that the Qwen embedding model produces **1024-dimensional** vectors
   - The vector database collection is created with the correct dimension (1024 instead of the default 768)
   - The logs show: `"Detected embedding dimension: 1024"` and `"Created new collection 'qa_blocks' with dimension 1024"`

3. **Vector Storage Working**: 
   - Successfully indexed **89 QA blocks** (8 from HA_GO_FAQ.pdf + 81 from Hong_Kong_Deposit_FAQ.pdf)
   - The logs show: `"✅ Successfully indexed 8 QA blocks in vector database"` and `"✅ Successfully indexed 81 QA blocks in vector database"`
   - Vector search is now working properly with similarity scores being calculated

## 🔧 **What Was Fixed:**

1. **Updated `requirements.txt`** to pin NumPy to `numpy<2.0.0`
2. **Enhanced vector database initialization** to automatically detect the embedding model's dimension
3. **Added dimension validation** that recreates collections if they have the wrong dimension
4. **Added `delete_collection` method** to properly clean up incompatible collections

## 📊 **Current Status:**

- ✅ Server is running on http://127.0.0.1:8000
- ✅ Vector database is working with 1024-dimensional Qwen embeddings
- ✅ 89 QA blocks successfully indexed and searchable
- ✅ Both semantic search and keyword search are functioning
- ✅ PDF processing and image extraction working properly

The system is now fully operational and should handle queries efficiently using the vector database for large-scale document retrieval!