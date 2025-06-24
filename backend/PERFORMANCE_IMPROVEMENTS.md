# Performance Improvements for Agentic RAG

This document outlines the performance improvements implemented to handle large vector datasets in the Agentic RAG system.

## Vector Database Integration

We've integrated FAISS (Facebook AI Similarity Search) to efficiently handle large vector datasets. This significantly improves performance when dealing with many PDF files.

### Key Benefits:

1. **Efficient Similarity Search**: FAISS uses Hierarchical Navigable Small World (HNSW) algorithm for approximate nearest neighbor search, which is much faster than linear search.
2. **Persistent Storage**: Vector embeddings are stored on disk, reducing memory usage.
3. **Metadata Filtering**: Supports filtering by metadata before or after vector search.
4. **Scalability**: Can handle millions of vectors efficiently.

## Implementation Details

### 1. Vector Database Service

A new `VectorDBService` class provides:

- **Collection Management**: Create, get, and rebuild vector collections
- **Vector Operations**: Add, search, and delete vectors
- **Metadata Management**: Store and retrieve metadata alongside vectors
- **Persistence**: Save and load indices from disk

```python
# Example: Searching the vector database
results = VectorDBService.search(
    collection_name="qa_blocks",
    query_vector=query_embedding,
    limit=top_n
)
```

### 2. Integration with Existing RAG Service

The vector database is integrated with minimal changes to existing business logic:

- **Fallback Mechanism**: If vector search fails, falls back to original in-memory search
- **Transparent Usage**: Same API for finding best matches regardless of backend
- **Graceful Degradation**: System works even if vector database is disabled

```python
# Seamless integration with existing code
if USE_VECTOR_DB:
    # Try vector database search
    # ...
else:
    # Fall back to original implementation
    # ...
```

### 3. API Endpoints for Management

New endpoints to manage the vector database:

- **GET /vector-db/status**: Check the status and statistics of the vector database
- **POST /vector-db/config**: Enable or disable the vector database
- **POST /vector-db/rebuild**: Rebuild the vector database (remove deleted vectors)

## Performance Comparison

| Scenario | Original Implementation | With Vector Database |
|----------|-------------------------|----------------------|
| 10 PDFs  | Fast                    | Fast                 |
| 100 PDFs | Slow                    | Fast                 |
| 1000 PDFs| Very Slow               | Fast                 |
| Memory Usage | High (all vectors in RAM) | Low (indices on disk) |
| Startup Time | Increases with # of PDFs | Constant |

## Usage Instructions

### Enabling/Disabling Vector Database

```bash
# Enable vector database
curl -X POST http://localhost:8000/vector-db/config -H "Content-Type: application/json" -d '{"enabled": true}'

# Disable vector database
curl -X POST http://localhost:8000/vector-db/config -H "Content-Type: application/json" -d '{"enabled": false}'
```

### Checking Vector Database Status

```bash
curl http://localhost:8000/vector-db/status
```

### Rebuilding Vector Database

If you've deleted many documents and want to reclaim space:

```bash
curl -X POST http://localhost:8000/vector-db/rebuild
```

## Technical Details

### Vector Index Structure

We use HNSW (Hierarchical Navigable Small World) index from FAISS, which:

- Creates a multi-layered graph structure for efficient navigation
- Provides logarithmic search complexity instead of linear
- Balances search speed and accuracy through configurable parameters

### Metadata Storage

Each vector is associated with metadata:

```json
{
  "id": "unique-id",
  "question": "Question text",
  "source_document": "document.pdf",
  "page_num": 5,
  "question_number": "Q11",
  "has_images": true,
  "deleted": false
}
```

### Caching Strategy

Multiple caching levels:

1. **PDF Parse Cache**: Avoid re-parsing the same PDF
2. **Embedding Cache**: Avoid re-computing embeddings for the same content
3. **Vector Database**: Persistent storage of vectors and metadata
4. **Query Results Cache**: Cache query results with TTL

## Future Improvements

1. **Vector Quantization**: Compress vectors to reduce storage and improve search speed
2. **Sharding**: Split the index across multiple files for very large collections
3. **Incremental Updates**: Update only changed vectors instead of rebuilding
4. **Hybrid Search**: Combine vector search with keyword search for better results 