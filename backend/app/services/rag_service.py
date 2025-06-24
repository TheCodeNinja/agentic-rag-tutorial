import fitz  # PyMuPDF
import base64
import re
from typing import List, Dict, Any, Generator
import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import torch
import json
import time
import uuid
import numpy as np
from .vector_db_service import VectorDBService
from langdetect import detect, LangDetectException

load_dotenv()

# --- Model and Global Variables ---

# Initialize the embedding model. Per user request, switching to a Qwen-based model.
# This model is chosen for its performance and because it's available from the HuggingFace model hub.
# Using 'mps' for Metal Performance Shaders on Apple Silicon for acceleration.
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# We'll initialize the model lazily or explicitly through initialize_embedding_model()
embedding_model = None

# A simple in-memory cache to avoid re-embedding the same content on every call.
# In a production system, you'd use a persistent vector database.
embedding_cache = {}

# Flag to indicate if we should use the vector database for search
USE_VECTOR_DB = True

# Collection name for QA blocks
QA_BLOCKS_COLLECTION = "qa_blocks"

Q_PATTERN = re.compile(r"^q\d*[\.:]", re.IGNORECASE)

def detect_chinese(text: str) -> bool:
    """
    Detect if the input text contains Chinese characters (both Traditional and Simplified).
    Returns True if Chinese characters are detected, False otherwise.
    
    Uses a combination of regex pattern matching and langdetect library for better accuracy.
    """
    # Unicode ranges for Chinese characters
    chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]+')
    
    # First check with regex for efficiency
    if chinese_pattern.search(text):
        return True
    
    # If no Chinese characters found with regex, try langdetect for more context
    try:
        lang = detect(text)
        return lang == 'zh-cn' or lang == 'zh-tw'
    except LangDetectException:
        # If langdetect fails, fall back to regex result (which was False)
        return False

def get_language_instruction(query: str) -> str:
    """
    Determine the language instruction for the LLM based on the query.
    If the query is in Chinese, instruct the LLM to respond in Chinese.
    Otherwise, respond in English.
    """
    if detect_chinese(query):
        return "Please respond in Chinese, using the same variant (Traditional or Simplified) as the query."
    return "Please respond in English."

def initialize_embedding_model():
    """Initialize the embedding model explicitly."""
    global embedding_model
    if embedding_model is None:
        print(f"Initializing embedding model on {DEVICE}...")
        start_time = time.time()
        embedding_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', device=DEVICE)
        print(f"Embedding model initialized in {time.time() - start_time:.2f} seconds")
    return embedding_model

def get_embedding_model():
    """Get or initialize the embedding model."""
    global embedding_model
    if embedding_model is None:
        initialize_embedding_model()
    return embedding_model

def initialize_vector_db():
    """Initialize the vector database."""
    # Create cache directories if they don't exist
    os.makedirs("cache/vector_indices", exist_ok=True)
    os.makedirs("cache/vector_metadata", exist_ok=True)
    
    # Initialize the vector database
    VectorDBService.initialize()
    
    # Get the embedding model to determine the correct dimension
    model = get_embedding_model()
    # Get dimension by encoding a sample text
    sample_embedding = model.encode("sample text")
    embedding_dim = len(sample_embedding)
    print(f"Detected embedding dimension: {embedding_dim}")
    
    # Create QA blocks collection with the correct dimension
    VectorDBService.get_or_create_collection(QA_BLOCKS_COLLECTION, dimension=embedding_dim)
    
    print(f"Vector database initialized with collection: {QA_BLOCKS_COLLECTION} (dimension: {embedding_dim})")

class QABlock:
    """Represents a Q&A block with associated text, images, and source document."""
    def __init__(self, question: str, page_num: int, y_start: float, source_document: str):
        self.id = str(uuid.uuid4())  # Add unique ID for vector database
        self.question = question
        self.source_document = source_document
        self.answer_text = ""
        self.images: List[str] = []
        self.page_num = page_num
        self.y_start = y_start
        self.end_page_num = -1
        self.y_end = float('inf')

    def add_text(self, text: str):
        self.answer_text += text + "\n"

    def add_image(self, img_base64: str):
        self.images.append(img_base64)

    def get_full_text(self) -> str:
        """Returns the text for embedding, without metadata."""
        return f"Question: {self.question}\nAnswer: {self.answer_text}"

    def get_text_for_llm(self) -> str:
        """Returns the text for the LLM, including source metadata."""
        return f"Source: {self.source_document} (Page {self.page_num + 1})\nQuestion: {self.question}\nAnswer: {self.answer_text}"

    def get_question_number(self) -> str:
        """Extract the question number (e.g., 'Q11') from the question text."""
        match = re.match(r'^(Q\d+)', self.question, re.IGNORECASE)
        return match.group(1).upper() if match else ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question, 
            "answer_text": self.answer_text, 
            "images": self.images, 
            "source_document": self.source_document,
            "page_num": self.page_num + 1,
            "question_number": self.get_question_number()
        }
        
    def to_metadata(self) -> Dict[str, Any]:
        """Convert to metadata for vector database."""
        return {
            "id": self.id,
            "question": self.question,
            "source_document": self.source_document,
            "page_num": self.page_num,
            "question_number": self.get_question_number(),
            "has_images": len(self.images) > 0,
            "deleted": False
        }

    def is_in_boundary(self, item_page_num: int, item_y: float) -> bool:
        """Checks if an item is within the QABlock's vertical and page boundaries."""
        is_after_start = (item_page_num > self.page_num) or \
                         (item_page_num == self.page_num and item_y > self.y_start)
        is_before_end = (item_page_num < self.end_page_num) or \
                        (item_page_num == self.end_page_num and item_y < self.y_end)
        return is_after_start and is_before_end

# We'll use a cache for parsed PDFs to avoid re-parsing the same file
pdf_parse_cache = {}

# Dictionary to map QA block IDs to actual QA blocks for quick lookup
qa_blocks_by_id = {}

def parse_pdf_into_qa_blocks(file_path: str, filename: str) -> List[QABlock]:
    """
    Parses a PDF using a multi-pass approach to deterministically identify Q&A blocks 
    and associate images based on their location relative to question anchors.
    """
    # Check if we've already parsed this file (using modification time as part of the cache key)
    mod_time = os.path.getmtime(file_path)
    cache_key = f"{file_path}_{mod_time}"
    if cache_key in pdf_parse_cache:
        return pdf_parse_cache[cache_key]
        
    try:
        doc = fitz.open(file_path)
        qa_blocks: List[QABlock] = []
        
        # Debug: Search for individual questions in the raw PDF text
        print(f"\n=== SEARCHING FOR QUESTIONS IN {filename} ===")
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            question_matches = re.findall(r'Q\d+', page_text)
            if question_matches:
                print(f"Page {page_num + 1}: Found questions {question_matches}")
        print("=== END QUESTION SEARCH ===\n")

        # 1. First Pass: Find all question anchors across all pages
        for page_num, page in enumerate(doc):
            text_blocks = page.get_text("dict", sort=True)["blocks"]
            for block in text_blocks:
                if block['type'] == 0:
                    block_text = "".join(span['text'] for line in block['lines'] for span in line['spans']).strip()
                    if Q_PATTERN.match(block_text):
                        print(f"Found Q block on page {page_num + 1}: '{block_text[:200]}...'")
                        # Check if this block contains multiple questions
                        if block_text.count('Q') > 1:
                            print(f"  --> This block contains multiple questions!")
                            # Try to split the block into individual questions
                            individual_questions = re.split(r'(?=Q\d+\.)', block_text)
                            for i, question in enumerate(individual_questions):
                                question = question.strip()
                                if question and Q_PATTERN.match(question):
                                    # Adjust Y position for subsequent questions in the same block
                                    y_offset = i * 20  # Approximate line height
                                    qa_blocks.append(QABlock(question=question, page_num=page_num, y_start=block['bbox'][1] + y_offset, source_document=filename))
                                    print(f"    Created individual block: '{question[:50]}...'")
                        else:
                            qa_blocks.append(QABlock(question=block_text, page_num=page_num, y_start=block['bbox'][1], source_document=filename))
        
        if not qa_blocks:
            doc.close()
            return []

        # 2. Second Pass: Define the end boundary for each Q&A block
        for i in range(len(qa_blocks) - 1):
            next_block = qa_blocks[i+1]
            qa_blocks[i].end_page_num = next_block.page_num
            qa_blocks[i].y_end = next_block.y_start
        qa_blocks[-1].end_page_num = len(doc) - 1
        qa_blocks[-1].y_end = doc[-1].rect.height

        # 3. Third Pass: Associate all content (text and images) with the correct block
        for page_num, page in enumerate(doc):
            try:
                # Associate text
                for block in page.get_text("dict")["blocks"]:
                    if block['type'] == 0:
                        block_text = "".join(span['text'] for line in block['lines'] for span in line['spans']).strip()
                        if not Q_PATTERN.match(block_text):
                            for qa in qa_blocks:
                                if qa.is_in_boundary(page_num, block['bbox'][1]):
                                    qa.add_text(block_text)
                                    break
            except Exception as e:
                print(f"Error processing text on page {page_num + 1}: {str(e)}")
                
            try:            
                # Associate images
                for img_info in page.get_images(full=True):
                    img_bbox = page.get_image_bbox(img_info)
                    img_middle_y = (img_bbox.y0 + img_bbox.y1) / 2
                    for qa in qa_blocks:
                        if qa.is_in_boundary(page_num, img_middle_y):
                            try:
                                xref = img_info[0]
                                base_image = doc.extract_image(xref)
                                img_base64 = base64.b64encode(base_image["image"]).decode('utf-8')
                                qa.add_image(f"data:image/{base_image['ext']};base64,{img_base64}")
                            except Exception as img_error:
                                print(f"Error extracting image on page {page_num + 1}: {str(img_error)}")
                            break
            except Exception as e:
                print(f"Error processing images on page {page_num + 1}: {str(e)}")
        
        doc.close()
        
        # Debug: Show all parsed questions
        print(f"\n=== PARSED QUESTIONS FROM {filename} ===")
        print(f"Total QA blocks found: {len(qa_blocks)}")
        for i, block in enumerate(qa_blocks):
            # Extract just the question number/identifier from the question text
            question_preview = block.question[:100].replace('\n', ' ')
            print(f"Block {i}: Page {block.page_num + 1} - '{question_preview}...'")
        print("=== END PARSED QUESTIONS ===\n")
        
        # Cache the result
        pdf_parse_cache[cache_key] = qa_blocks
        
        # If vector database is enabled, add the blocks to the vector database
        if USE_VECTOR_DB:
            index_qa_blocks(qa_blocks)
        
        # Update the ID-to-block mapping
        for block in qa_blocks:
            qa_blocks_by_id[block.id] = block
        
        return qa_blocks
    except Exception as e:
        print(f"Error parsing PDF {filename}: {str(e)}")
        return []

def index_qa_blocks(qa_blocks: List[QABlock]):
    """Index QA blocks in the vector database."""
    if not USE_VECTOR_DB:
        return
    
    print(f"Indexing {len(qa_blocks)} QA blocks in vector database...")
    
    # Get the embedding model
    model = get_embedding_model()
    
    # Prepare vectors for batch indexing
    vectors_to_add = []
    
    for block in qa_blocks:
        # Check if we already have an embedding for this block
        cache_key = (block.source_document, block.question)
        if cache_key not in embedding_cache:
            # Generate embedding for this block
            embedding = model.encode(block.get_full_text(), convert_to_tensor=True, device=DEVICE)
            embedding_cache[cache_key] = embedding
        else:
            embedding = embedding_cache[cache_key]
        
        # Add to vectors list
        vectors_to_add.append((
            block.id,
            embedding,
            block.to_metadata()
        ))
    
    # Add vectors to the vector database
    if vectors_to_add:
        try:
            VectorDBService.add_vectors(QA_BLOCKS_COLLECTION, vectors_to_add)
            print(f"✅ Successfully indexed {len(vectors_to_add)} QA blocks in vector database")
        except Exception as e:
            print(f"❌ Error indexing QA blocks in vector database: {str(e)}")

def find_images_for_question_number(question_number: str, all_qa_blocks: List[QABlock]) -> List[str]:
    """
    Find images that belong to a specific question number (e.g., 'Q11').
    This handles cases where multiple questions are grouped in one block.
    """
    question_number = question_number.upper()
    
    # First, try to find an exact match for the question number
    for block in all_qa_blocks:
        if block.get_question_number().upper() == question_number:
            print(f"Found exact match for {question_number}: {len(block.images)} images")
            return block.images
    
    # If no exact match, look for blocks that contain this question
    for block in all_qa_blocks:
        if question_number in block.question.upper():
            print(f"Found {question_number} within block: '{block.question[:100]}...'")
            # If this block contains multiple questions, we need to be more careful about images
            # For now, return all images from the block (could be refined further)
            return block.images
    
    print(f"No images found for question {question_number}")
    return []

def find_best_matches(query: str, qa_blocks: List[QABlock], top_n: int = 3) -> List[QABlock]:
    """
    Finds the most relevant QABlocks using vector embeddings and cosine similarity.
    If vector database is enabled, it uses that for efficient search.
    Otherwise, falls back to the original implementation.
    """
    if not qa_blocks:
        return []
    
    # Debug: Find QA blocks that might be relevant based on keywords
    relevant_keywords = ["document", "submit", "gopc", "sopc", "booking", "application", "prepare"]
    print(f"\n=== KEYWORD SEARCH DEBUG ===")
    print(f"Looking for QA blocks containing keywords: {relevant_keywords}")
    keyword_matches = []
    for i, block in enumerate(qa_blocks):
        # Convert both text and keywords to lowercase for case-insensitive matching
        block_text_lower = (block.question + " " + block.answer_text).lower()
        matching_keywords = [kw for kw in relevant_keywords if kw.lower() in block_text_lower]
        if matching_keywords:
            keyword_matches.append((i, block, matching_keywords))
    
    print(f"Found {len(keyword_matches)} blocks with relevant keywords:")
    for i, (block_idx, block, keywords) in enumerate(keyword_matches[:10]):  # Show top 10
        print(f"  {i+1}. Block {block_idx}: '{block.question}' - Keywords: {keywords}")
        print(f"     Source: {block.source_document}, Page: {block.page_num + 1}")
    print("=== END KEYWORD SEARCH DEBUG ===\n")
    
    # 1. Generate embedding for the user's query, using a prompt for better retrieval performance.
    model = get_embedding_model()
    
    # Enhance query with uppercase variants of common medical terms for better matching
    enhanced_query = query
    medical_terms = ["sopc", "gopc", "ha"]
    for term in medical_terms:
        if term.lower() in query.lower() and term.lower() not in query:
            # Add uppercase variant if the lowercase version is in the query
            enhanced_query += f" {term.upper()}"
    
    # Log the enhanced query if it was modified
    if enhanced_query != query:
        print(f"Enhanced query for better matching: '{query}' → '{enhanced_query}'")
    
    query_embedding = model.encode(enhanced_query, convert_to_tensor=True, device=DEVICE, prompt_name="query")
    
    # If vector database is enabled, use it for search
    if USE_VECTOR_DB:
        try:
            # Search the vector database
            results = VectorDBService.search(QA_BLOCKS_COLLECTION, query_embedding, limit=top_n)
            
            # Map results to QA blocks
            best_blocks = []
            for result in results:
                block_id = result['id']
                if block_id in qa_blocks_by_id:
                    best_blocks.append(qa_blocks_by_id[block_id])
                    
                    # Log the result
                    block = qa_blocks_by_id[block_id]
                    print(f"RANK {len(best_blocks)}: Similarity Score = {result['score']:.6f}")
                    print(f"  Block ID: {block_id}")
                    print(f"  Question: '{block.question}'")
                    print(f"  Source Document: '{block.source_document}'")
                    print(f"  Page Number: {block.page_num + 1}")
                    print(f"  Number of Images: {len(block.images)}")
            
            if best_blocks:
                return best_blocks
            else:
                print("No results from vector database, falling back to in-memory search")
        except Exception as e:
            print(f"Error searching vector database: {str(e)}")
            print("Falling back to in-memory search")
    
    # Fall back to the original implementation if vector database search failed or is disabled

    # 2. Get or create embeddings for all QA blocks
    block_embeddings = []
    for block in qa_blocks:
        # Use a unique key for caching based on document and question
        cache_key = (block.source_document, block.question)
        if cache_key not in embedding_cache:
            # The text to be embedded includes the question and answer for better semantic context.
            # Documents are encoded without a prompt.
            embedding_cache[cache_key] = model.encode(block.get_full_text(), convert_to_tensor=True, device=DEVICE)
        block_embeddings.append(embedding_cache[cache_key])

    if not block_embeddings:
        return []
        
    # 3. Compute cosine similarity between query and all blocks
    # The result is a tensor of similarity scores.
    cosine_scores = util.cos_sim(query_embedding, torch.stack(block_embeddings))[0]

    # 4. Find the top N best matches based on score
    # We pair each block with its score, sort them, and take the top N.
    top_results = torch.topk(cosine_scores, k=min(top_n, len(qa_blocks)))
    
    # Log the top matches with their similarity scores
    print(f"\n=== DETAILED SIMILARITY SEARCH RESULTS for query: '{enhanced_query}' ===")
    print(f"Total QA blocks available: {len(qa_blocks)}")
    print(f"Requested top_n: {top_n}")
    print(f"Actual results returned: {len(top_results.indices)}")
    print("---")
    
    for i, (score, idx) in enumerate(zip(top_results.values, top_results.indices)):
        block = qa_blocks[idx]
        print(f"RANK {i+1}: Similarity Score = {score:.6f}")
        print(f"  Block Index: {idx}")
        print(f"  Question: '{block.question}'")
        print(f"  Source Document: '{block.source_document}'")
        print(f"  Page Number: {block.page_num + 1} (0-indexed: {block.page_num})")
        print(f"  Y-coordinates: start={block.y_start:.2f}, end={block.y_end:.2f}")
        print(f"  End Page: {block.end_page_num + 1} (0-indexed: {block.end_page_num})")
        print(f"  Answer Text Length: {len(block.answer_text)} characters")
        print(f"  Answer Preview: '{block.answer_text[:150]}...'" if len(block.answer_text) > 150 else f"  Full Answer: '{block.answer_text}'")
        print(f"  Number of Images: {len(block.images)}")
        print(f"  Full Text for Embedding: '{block.get_full_text()[:200]}...'" if len(block.get_full_text()) > 200 else f"  Full Text for Embedding: '{block.get_full_text()}'")
        print("  " + "="*80)
    print("=== END DETAILED SIMILARITY SEARCH RESULTS ===\n")
    
    # Return the top N blocks without a hard similarity threshold.
    # This allows the LLM to make the final determination of relevance.
    best_blocks = [qa_blocks[idx] for idx in top_results.indices]
    
    return best_blocks

def get_streaming_answer_from_llm(query: str, context_blocks: List[QABlock]) -> Generator[str, None, None]:
    """
    Gets a streaming answer from the LLM using the provided context blocks.
    Returns a generator that yields strings as they're received from the LLM.
    """
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    
    # Format the source documents as context for the prompt
    context_text = "\n\n".join([block.get_text_for_llm() for block in context_blocks])
    
    # Determine language instruction based on query
    language_instruction = get_language_instruction(query)
    
    # Construct a prompt that guides the LLM to answer based only on the sources
    system_message = f"""You are an agentic RAG assistant that answers questions based solely on the provided knowledge base sources.
For each user query, analyze the provided context documents and generate a helpful, accurate response.
Cite specific questions from the knowledge base using their identifier (e.g., "Q1", "Q5").
If the answer cannot be found within the provided context, acknowledge this limitation and do not make up information.
Do not mention the retrieval process or embedding mechanisms in your response.
{language_instruction}"""

    user_prompt = f"""I need information about the following query:
{query}

Here are the relevant sections from the knowledge base:

{context_text}

Please provide a comprehensive answer based solely on the provided information. 
If the answer is not clearly found in the provided context, please say so.

Organize your answer with clear structure and cite the relevant question numbers from the knowledge base."""

    # Call the OpenAI API with streaming enabled
    try:
        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            stream=True,
            temperature=0.5,
            max_tokens=800,
        )
        
        # Yield chunks as they arrive
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        print(f"Error in streaming response: {e}")
        yield f"Sorry, there was an error generating the response: {str(e)}"

def get_agentic_answer_from_llm(query: str, context_blocks: List[QABlock]) -> Dict[str, Any]:
    """
    Gets a synthesized answer from the LLM based on the provided context blocks,
    including source attribution and related processed blocks.
    """
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

    # Combine the text from all context blocks using the dedicated method for LLMs
    context = "\n\n---\n\n".join([block.get_text_for_llm() for block in context_blocks])
    
    # Determine language instruction based on query
    language_instruction = get_language_instruction(query)
    
    # Log the exact context being sent to the LLM
    print(f"\n=== CONTEXT SENT TO LLM ===")
    print(f"Query: '{query}'")
    print(f"Number of context blocks: {len(context_blocks)}")
    for i, block in enumerate(context_blocks):
        print(f"Block {i+1}:")
        print(f"  Question: '{block.question}'")
        print(f"  Source: {block.source_document}")
        print(f"  Page (0-indexed): {block.page_num}")
        print(f"  Page (1-indexed): {block.page_num + 1}")
        print(f"  Text for LLM: '{block.get_text_for_llm()[:300]}...'")
        print("  ---")
    print("=== END CONTEXT TO LLM ===\n")

    prompt = f"""
    You are an advanced research agent. Your task is to synthesize a single, comprehensive answer to the user's question based on one or more provided context snippets from different documents.

    **Instructions:**
    1.  Carefully analyze the user's question and all the provided context snippets.
    2.  Formulate a single, cohesive, and natural-sounding answer. Use Markdown formatting (like bullet points, bolding, and italics) to improve readability.
    3.  Your answer MUST be based exclusively on the information found in the provided snippets. Do not add outside information.
    4.  After formulating the answer, you MUST identify the exact source snippets you used. For each source you used, provide the exact `Question:` text and `Source:` information as they appear in the context snippets.
    5.  IMPORTANT: When extracting page numbers, use the page number that appears in the "Source:" line of each context snippet. These page numbers are already human-readable (1-indexed), not computer-indexed (0-indexed).
    6.  {language_instruction}
    7.  Return a single JSON object with two keys:
        -   `llm_answer`: A string containing your synthesized answer.
        -   `sources`: A JSON list of objects, where each object has three keys: `question` (the exact question text from the source snippet), `source_document` (the filename), and `page_num` (the page number exactly as it appears in the Source line - this should be 1-indexed).

    **Context Snippets:**
    ---
    {context}
    ---

    **User's Question:** {query}

    **JSON Response:**
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a research agent that synthesizes answers from multiple sources and provides a response in JSON format."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    
    try:
        # For simplicity, we'll just return the parsed JSON from the model.
        # In a production system, you'd add validation here.
        result = json.loads(response.choices[0].message.content)
        
        # Log what the LLM returned as sources
        print(f"\n=== LLM RESPONSE SOURCES ===")
        print(f"LLM Answer: {result.get('llm_answer', 'No answer')[:100]}...")
        print(f"LLM Sources: {result.get('sources', [])}")
        
        # Debug: Cross-reference LLM sources with original context blocks
        print(f"\n=== SOURCE VALIDATION ===")
        llm_sources = result.get('sources', [])
        for i, llm_source in enumerate(llm_sources):
            print(f"LLM Source {i+1}:")
            print(f"  LLM says: Question='{llm_source.get('question', 'N/A')}'")
            print(f"  LLM says: Document='{llm_source.get('source_document', 'N/A')}'")
            print(f"  LLM says: Page={llm_source.get('page_num', 'N/A')}")
            
            # Try to find this source in our original context blocks
            found_match = False
            for j, block in enumerate(context_blocks):
                if (llm_source.get('source_document') == block.source_document and
                    str(llm_source.get('page_num')) == str(block.page_num + 1)):
                    print(f"  ✓ MATCHES Context Block {j+1}: '{block.question}'")
                    found_match = True
                    break
            
            if not found_match:
                print(f"  ✗ NO MATCH found in context blocks!")
                print(f"  Available context blocks:")
                for j, block in enumerate(context_blocks):
                    print(f"    Block {j+1}: {block.source_document} Page {block.page_num + 1} - '{block.question[:50]}...'")
        print("=== END SOURCE VALIDATION ===\n")
        print("=== END LLM RESPONSE ===\n")
        
        return result
    except (json.JSONDecodeError, IndexError):
        return {"llm_answer": "There was an error processing the response from the AI.", "sources": []}

def get_cot_agentic_answer(query: str, qa_blocks: List[QABlock], top_n: int = 3) -> Dict[str, Any]:
    """
    Implements a chain-of-thought approach where the agent first plans what information
    it needs before retrieval, then performs the retrieval, and finally synthesizes the answer.
    
    This function preserves all existing business logic but adds a planning step before retrieval.
    """
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    
    # Determine language instruction based on query
    language_instruction = get_language_instruction(query)
    
    # Step 1: Planning - Ask the LLM to analyze the query and plan what information it needs
    planning_prompt = f"""
    You are an advanced research agent. Your task is to analyze a user's question and develop a plan to answer it.
    
    **User's Question:** {query}
    
    Please follow these steps:
    1. Analyze the question to identify the key concepts, entities, and relationships.
    2. Break down the question into sub-questions or information needs.
    3. Formulate a plan for what specific information you would need to fully answer this question.
    4. Identify any potential keywords or terms that would be important for retrieving relevant information.
    
    {language_instruction}
    
    Return your analysis as a JSON object with the following structure:
    {{
        "key_concepts": ["list", "of", "key", "concepts"],
        "sub_questions": ["list", "of", "specific", "sub-questions"],
        "information_needs": ["list", "of", "specific", "information", "needed"],
        "search_keywords": ["list", "of", "important", "search", "terms"]
    }}
    """
    
    # Get the planning response
    planning_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a research planning agent that analyzes questions and creates structured information retrieval plans."},
            {"role": "user", "content": planning_prompt}
        ],
        max_tokens=500,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    
    try:
        # Parse the planning response
        plan = json.loads(planning_response.choices[0].message.content)
        
        # Log the planning process
        print(f"\n=== CHAIN OF THOUGHT PLANNING ===")
        print(f"Original Query: '{query}'")
        print(f"Key Concepts: {plan.get('key_concepts', [])}")
        print(f"Sub-questions: {plan.get('sub_questions', [])}")
        print(f"Information Needs: {plan.get('information_needs', [])}")
        print(f"Search Keywords: {plan.get('search_keywords', [])}")
        print("=== END PLANNING ===\n")
        
        # Step 2: Enhanced Retrieval - Use the planning information to improve retrieval
        # We'll use the original find_best_matches function but with enhanced query information
        
        # Create an enhanced query by combining the original query with key search terms
        search_keywords = plan.get('search_keywords', [])
        enhanced_query = query
        if search_keywords:
            # Add the top 3 keywords to the query if they're not already present
            for keyword in search_keywords[:3]:
                if keyword.lower() not in query.lower():
                    enhanced_query += f" {keyword}"
        
        print(f"Enhanced Query for Retrieval: '{enhanced_query}'")
        
        # Use the existing retrieval mechanism with the enhanced query
        best_matches = find_best_matches(enhanced_query, qa_blocks, top_n)
        
        # Step 3: Answer Synthesis with Chain-of-Thought
        # Combine the text from all context blocks
        context = "\n\n---\n\n".join([block.get_text_for_llm() for block in best_matches])
        
        # Create a prompt that includes the planning information and asks for chain-of-thought reasoning
        cot_prompt = f"""
        You are an advanced research agent. Your task is to synthesize a comprehensive answer to the user's question based on provided context snippets and your analysis plan.
        
        **User's Question:** {query}
        
        **Your Analysis Plan:**
        - Key Concepts: {', '.join(plan.get('key_concepts', []))}
        - Sub-questions: {', '.join(plan.get('sub_questions', []))}
        - Information Needs: {', '.join(plan.get('information_needs', []))}
        
        **Context Snippets:**
        ---
        {context}
        ---
        
        {language_instruction}
        
        Please follow these steps in your reasoning process:
        1. First, examine each context snippet and identify which parts address your sub-questions or information needs.
        2. For each sub-question or information need, extract the relevant information from the context snippets.
        3. Synthesize this information into a cohesive answer that addresses the original question.
        4. Note any sub-questions or information needs that couldn't be addressed by the provided context.
        
        Return a JSON object with the following structure:
        {{
            "reasoning_process": "A detailed explanation of your step-by-step reasoning process, showing how you connected information from different sources",
            "llm_answer": "The final synthesized answer to the user's question, using Markdown formatting for readability",
            "sources": [
                {{
                    "question": "The exact question text from the source snippet",
                    "source_document": "The filename of the source document",
                    "page_num": "The page number as it appears in the Source line (1-indexed)"
                }}
            ],
            "unanswered_aspects": ["Any sub-questions or aspects that couldn't be answered from the provided context"]
        }}
        """
        
        # Get the chain-of-thought response
        cot_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a research agent that uses chain-of-thought reasoning to synthesize answers from multiple sources."},
                {"role": "user", "content": cot_prompt}
            ],
            max_tokens=1500,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        
        # Parse and return the final response
        result = json.loads(cot_response.choices[0].message.content)
        
        # Log the reasoning process
        print(f"\n=== CHAIN OF THOUGHT REASONING ===")
        print(f"Reasoning Process: {result.get('reasoning_process', 'No reasoning provided')[:300]}...")
        print(f"Unanswered Aspects: {result.get('unanswered_aspects', [])}")
        print("=== END REASONING ===\n")
        
        # Include the original retrieved content for context
        result["retrieved_context"] = [block.to_dict() for block in best_matches]
        
        # Add question-specific images to each source in the LLM response
        if "sources" in result:
            for source in result["sources"]:
                # Extract question number from the LLM's source
                question_text = source.get("question", "")
                match = re.match(r'^(Q\d+)', question_text, re.IGNORECASE)
                if match:
                    question_number = match.group(1).upper()
                    print(f"Looking for images for {question_number} from LLM source")
                    # Find images for this specific question number
                    question_images = find_images_for_question_number(question_number, qa_blocks)
                    source["images"] = question_images
                    print(f"Added {len(question_images)} images for {question_number}")
                else:
                    source["images"] = []
        
        return result
    except (json.JSONDecodeError, IndexError) as e:
        print(f"Error in chain-of-thought processing: {e}")
        return {"llm_answer": "There was an error processing the response from the AI.", "sources": []} 