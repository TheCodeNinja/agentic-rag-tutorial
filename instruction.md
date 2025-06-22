**Task Objective:**  
Build an agentic Retrieval-Augmented Generation (RAG) application leveraging the DeepSeek Large Language Model (LLM) to accurately handle PDF-based FAQs containing both textual explanations and images. The application will have a clearly separated architecture consisting of:

- A backend REST API built with **FastAPI**.
- A frontend user interface built using **React with TypeScript**.

---

## Detailed Requirements:

### 1. Backend (FastAPI):

- **Framework & Language:** Python with FastAPI.
- **Core Functionality:**
  - Implement agentic RAG to query and retrieve relevant textual content and images from PDF FAQs based on user questions.
  - Integrate DeepSeek LLM for processing and generating accurate and contextually relevant responses.
  - Provide RESTful endpoints that:
    - Accept user queries from the frontend.
    - Retrieve and process data from PDF documents.
    - Return structured JSON responses containing:
      - Generated textual answers.
      - References to relevant extracted text snippets.
      - Associated image data (encoded as base64 or URLs to temporary storage).
- **Performance and Scalability:**
  - Optimize response times and ensure scalability for concurrent users.
  - Clearly document endpoints using FastAPI's built-in interactive docs (Swagger/OpenAPI).
- **Error Handling & Validation:**
  - Implement comprehensive error handling for invalid inputs, server-side issues, and retrieval errors.
  - Utilize appropriate HTTP status codes and descriptive error messages.

---

### 2. Frontend (React TypeScript):

- **Framework & Language:** React with TypeScript.
- **Core Functionality:**
  - Provide an intuitive, responsive user interface for users to ask questions.
  - Communicate asynchronously with the FastAPI backend using HTTP requests (e.g., Axios or Fetch API).
  - Clearly display generated textual answers alongside relevant images retrieved from the backend.
- **UI Requirements:**
  - Implement form input for user queries with validation.
  - Display backend-generated answers clearly, ensuring readability and visual coherence.
  - Responsively render images alongside relevant text content.
- **State Management:**
  - Utilize modern React features (Hooks, Context API) or state management libraries (e.g., Zustand, Redux) as preferred.
- **Accessibility and User Experience:**
  - Ensure the application follows accessibility best practices.
  - Prioritize a clean, professional, and intuitive user experience (UX).

---

## Example Workflow:

1. **Frontend Input:** User submits a question via the React interface (e.g., "How do I reset my account password?").
2. **Backend Processing (FastAPI):** 
   - Backend endpoint receives the query.
   - Agentic retrieval system extracts relevant text and images from PDF FAQs.
   - DeepSeek LLM synthesizes a concise, accurate answer.
   - Backend returns structured JSON response including:
     - Generated answer text.
     - Relevant textual excerpts used.
     - Base64-encoded images or URLs referencing images.
3. **Frontend Display:**
   - React app receives the backend response.
   - Displays the generated answer clearly.
   - Renders associated images seamlessly alongside the textual content.

---

### Expected Outcome:

The resulting application should deliver a seamless user experience, combining the strengths of DeepSeek LLM, agentic RAG techniques, robust FastAPI backend, and an interactive React TypeScript frontend.