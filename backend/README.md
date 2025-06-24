# Agentic RAG with Chain-of-Thought Reasoning and Vector Database

This project implements an Agentic RAG (Retrieval-Augmented Generation) system with enhanced Chain-of-Thought reasoning capability and efficient vector database integration for handling large document collections.

## Chain-of-Thought Implementation

The Chain-of-Thought (CoT) feature has been added without modifying the existing business logic. It follows a three-step process:

### 1. Planning Phase

Before retrieving information, the LLM analyzes the query to:
- Identify key concepts and entities
- Break down the question into sub-questions
- Formulate a plan for what information is needed
- Identify important search keywords

```json
{
  "key_concepts": ["concept1", "concept2"],
  "sub_questions": ["What is X?", "How does Y work?"],
  "information_needs": ["Definition of X", "Process of Y"],
  "search_keywords": ["keyword1", "keyword2"]
}
```

### 2. Enhanced Retrieval

The system uses the planning information to improve retrieval:
- Augments the original query with key search terms
- Uses the existing semantic search mechanism with the enhanced query
- Preserves all existing business logic for document retrieval

### 3. Reasoning-Based Answer Synthesis

The LLM generates a response using explicit reasoning steps:
- Examines each context snippet to identify relevant information
- Maps information to the sub-questions identified in planning
- Synthesizes a cohesive answer addressing the original query
- Notes any sub-questions that couldn't be addressed by the provided context

```json
{
  "reasoning_process": "Step-by-step explanation of how information was connected",
  "llm_answer": "The synthesized answer to the user's question",
  "sources": [{"question": "Q1", "source_document": "doc.pdf", "page_num": 5}],
  "unanswered_aspects": ["Questions that couldn't be answered from context"]
}
```

## Vector Database Integration

To efficiently handle large document collections, the system now includes FAISS vector database integration:

### Key Benefits:

1. **Efficient Similarity Search**: Uses HNSW algorithm for fast approximate nearest neighbor search
2. **Persistent Storage**: Vector embeddings are stored on disk, reducing memory usage
3. **Graceful Fallback**: Falls back to in-memory search if vector database is unavailable
4. **Scalability**: Can handle millions of vectors efficiently

### Usage:

The vector database can be enabled/disabled via API:

```bash
# Enable vector database
curl -X POST http://localhost:8000/vector-db/config -H "Content-Type: application/json" -d '{"enabled": true}'

# Check status
curl http://localhost:8000/vector-db/status
```

For more details on the vector database implementation, see [PERFORMANCE_IMPROVEMENTS.md](PERFORMANCE_IMPROVEMENTS.md).

## API Endpoints

- **Regular RAG**: `/ask` - Uses the standard RAG approach
- **Chain-of-Thought RAG**: `/ask/cot` - Uses the enhanced CoT approach
- **Vector DB Status**: `/vector-db/status` - Check vector database status
- **Vector DB Config**: `/vector-db/config` - Enable/disable vector database
- **Vector DB Rebuild**: `/vector-db/rebuild` - Rebuild vector database indices

## Frontend Integration

The frontend includes:
- Toggle switch to enable/disable Chain-of-Thought reasoning
- Display of reasoning process and unanswered aspects when CoT is enabled

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload
```

## Benefits

1. **Transparency**: Users can see the reasoning process behind answers
2. **Improved Retrieval**: Enhanced query formulation leads to better document retrieval
3. **Gap Identification**: Explicitly identifies what aspects couldn't be answered
4. **Scalability**: Efficiently handles large document collections
5. **No Disruption**: Implemented without changing existing business logic 