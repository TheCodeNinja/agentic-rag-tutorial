I'll walk through the Agentic RAG workflow using sample data examples to illustrate each step in detail.
Ran tool
Let me create a step-by-step explanation with hypothetical sample data since there are no actual PDF files in the data directory:

# Agentic RAG Workflow with Sample Data

## Step 1: PDF Document Processing

**Sample PDF**: `healthcare_faq.pdf` containing questions and answers about healthcare services

```
Q1: How do I schedule a GOPC appointment?
To schedule a General Outpatient Clinic (GOPC) appointment, you can:
• Use the HA Go mobile app
• Call the booking hotline at 1234-5678
• Visit the clinic in person with your HKID

[IMAGE: Screenshot of HA Go app booking interface]

Q2: What documents should I bring to my appointment?
Please bring the following documents:
• Hong Kong ID Card or valid identification
• Appointment slip (if applicable)
• Medical records from previous visits
• Current medication list

[IMAGE: Photo of sample documents arranged neatly]
```

### Processing:
1. **First Pass**: System identifies "Q1" and "Q2" as question anchors
2. **Second Pass**: Sets Q1's boundary from its y-position until Q2's y-position
3. **Third Pass**: Associates text and images with each QA block

### Result:
```python
qa_blocks = [
    QABlock(
        question="Q1: How do I schedule a GOPC appointment?",
        answer_text="To schedule a General Outpatient Clinic (GOPC) appointment, you can:\n• Use the HA Go mobile app\n• Call the booking hotline at 1234-5678\n• Visit the clinic in person with your HKID",
        images=["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."],  # Base64 of HA Go app screenshot
        source_document="healthcare_faq.pdf",
        page_num=0
    ),
    QABlock(
        question="Q2: What documents should I bring to my appointment?",
        answer_text="Please bring the following documents:\n• Hong Kong ID Card or valid identification\n• Appointment slip (if applicable)\n• Medical records from previous visits\n• Current medication list",
        images=["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."],  # Base64 of documents photo
        source_document="healthcare_faq.pdf",
        page_num=0
    )
]
```

## Step 2: Vector Embedding and Semantic Search

**User Query**: "How can I book a GOPC appointment?"

### Processing:
1. **Query Embedding**: 
   ```python
   query_embedding = embedding_model.encode(
       "How can I book a GOPC appointment?", 
       convert_to_tensor=True, 
       device=DEVICE, 
       prompt_name="query"
   )
   # Result: [0.123, -0.456, 0.789, ...] (vector of 768 dimensions)
   ```

2. **Document Embeddings**:
   ```python
   # For QA Block 1
   block1_text = "Question: Q1: How do I schedule a GOPC appointment?\nAnswer: To schedule a General Outpatient Clinic (GOPC) appointment, you can:\n• Use the HA Go mobile app\n• Call the booking hotline at 1234-5678\n• Visit the clinic in person with your HKID"
   block1_embedding = embedding_model.encode(block1_text, convert_to_tensor=True, device=DEVICE)
   # Result: [0.111, -0.222, 0.333, ...] (vector of 768 dimensions)
   
   # For QA Block 2
   block2_text = "Question: Q2: What documents should I bring to my appointment?\nAnswer: Please bring the following documents:\n• Hong Kong ID Card or valid identification\n• Appointment slip (if applicable)\n• Medical records from previous visits\n• Current medication list"
   block2_embedding = embedding_model.encode(block2_text, convert_to_tensor=True, device=DEVICE)
   # Result: [0.444, -0.555, 0.666, ...] (vector of 768 dimensions)
   ```

3. **Similarity Calculation**:
   ```python
   # Cosine similarity between query and block1
   similarity1 = util.cos_sim(query_embedding, block1_embedding)[0]
   # Result: tensor([0.89])  (high similarity)
   
   # Cosine similarity between query and block2
   similarity2 = util.cos_sim(query_embedding, block2_embedding)[0]
   # Result: tensor([0.32])  (lower similarity)
   ```

4. **Results Ranking**:
   ```python
   # Ranked results (indices and scores)
   indices = [0, 1]  # Block indices
   scores = [0.89, 0.32]  # Similarity scores
   
   # Best matches (QA blocks ranked by relevance)
   best_matches = [qa_blocks[0]]  # Only returning top 1 result
   ```

## Step 3: Caching in Action

### Embedding Cache:
```python
# First query
cache_key = ("healthcare_faq.pdf", "Q1: How do I schedule a GOPC appointment?")
embedding_cache[cache_key] = block1_embedding

# Later, when the same block needs embedding
if cache_key in embedding_cache:
    # Reuse cached embedding instead of recomputing
    embedding = embedding_cache[cache_key]
```

### Context Blocks Cache:
```python
# After processing a query
context_blocks_cache["How can I book a GOPC appointment?"] = {
    "all_qa_blocks": qa_blocks,
    "best_matches": best_matches,
    "timestamp": 1652345678.123
}

# Later, for the same query
if "How can I book a GOPC appointment?" in context_blocks_cache:
    cache_entry = context_blocks_cache["How can I book a GOPC appointment?"]
    if time.time() - cache_entry["timestamp"] < 300:  # 5 minutes TTL
        # Reuse cached results
        all_qa_blocks = cache_entry["all_qa_blocks"]
        best_matches = cache_entry["best_matches"]
```

## Step 4: Agentic LLM Response Generation

### Input to LLM:
```
You are an advanced research agent. Your task is to synthesize a single, comprehensive answer to the user's question based on one or more provided context snippets from different documents.

**Context Snippets:**
---
Source: healthcare_faq.pdf (Page 1)
Question: Q1: How do I schedule a GOPC appointment?
Answer: To schedule a General Outpatient Clinic (GOPC) appointment, you can:
• Use the HA Go mobile app
• Call the booking hotline at 1234-5678
• Visit the clinic in person with your HKID
---

**User's Question:** How can I book a GOPC appointment?
```

### LLM Response (JSON):
```json
{
  "llm_answer": "You can book a General Outpatient Clinic (GOPC) appointment through three methods:\n\n• **Mobile App**: Use the HA Go mobile application\n• **Phone**: Call the booking hotline at 1234-5678\n• **In Person**: Visit the clinic directly with your HKID card",
  "sources": [
    {
      "question": "Q1: How do I schedule a GOPC appointment?",
      "source_document": "healthcare_faq.pdf",
      "page_num": 1
    }
  ]
}
```

## Step 5: Image Association for Response

### Processing:
1. **Extract Question Number**:
   ```python
   question_text = "Q1: How do I schedule a GOPC appointment?"
   match = re.match(r'^(Q\d+)', question_text, re.IGNORECASE)
   question_number = match.group(1).upper()  # "Q1"
   ```

2. **Find Images for Question**:
   ```python
   # Look for QA blocks with this question number
   for block in qa_blocks:
       if block.get_question_number() == "Q1":
           question_images = block.images
           # ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."]
   ```

3. **Attach Images to Response**:
   ```python
   # Add images to the source in the LLM response
   for source in agentic_response["sources"]:
       if source["question"].startswith("Q1:"):
           source["images"] = question_images
   ```

### Final Response with Images:
```json
{
  "llm_answer": "You can book a General Outpatient Clinic (GOPC) appointment through three methods:\n\n• **Mobile App**: Use the HA Go mobile application\n• **Phone**: Call the booking hotline at 1234-5678\n• **In Person**: Visit the clinic directly with your HKID card",
  "sources": [
    {
      "question": "Q1: How do I schedule a GOPC appointment?",
      "source_document": "healthcare_faq.pdf",
      "page_num": 1,
      "images": ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."]
    }
  ],
  "retrieved_context": [
    {
      "question": "Q1: How do I schedule a GOPC appointment?",
      "answer_text": "To schedule a General Outpatient Clinic (GOPC) appointment, you can:\n• Use the HA Go mobile app\n• Call the booking hotline at 1234-5678\n• Visit the clinic in person with your HKID",
      "images": ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."],
      "source_document": "healthcare_faq.pdf",
      "page_num": 1,
      "question_number": "Q1"
    }
  ]
}
```

## Step 6: System Warm-up in Action

During server startup:

```python
# Initialize embedding model
embedding_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', device=DEVICE)

# Pre-load PDFs
pdf_files = ["healthcare_faq.pdf", "medical_services.pdf"]
all_blocks = []

# Process each PDF
for filename in pdf_files:
    file_path = os.path.join(DATA_DIR, filename)
    blocks = parse_pdf_into_qa_blocks(file_path, filename)
    all_blocks.extend(blocks)
    print(f"✅ Successfully processed {filename}, found {len(blocks)} QA blocks")

# Store in global cache
preloaded_qa_blocks = all_blocks
print(f"🎉 System warm-up complete! Pre-loaded {len(preloaded_qa_blocks)} QA blocks from {len(pdf_files)} PDFs")
```

This detailed walkthrough with sample data demonstrates how the Agentic RAG system processes documents, performs semantic search, generates responses, and associates relevant images to provide comprehensive, context-aware answers to user queries.