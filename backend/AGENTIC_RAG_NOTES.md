Yes, your project is indeed an agentic RAG (Retrieval-Augmented Generation) system, though with a moderate level of agency. Here's why it qualifies:

### Agentic Elements Present:

1. **Semantic Search and Retrieval**: Your system uses advanced embedding models (Qwen) to find the most semantically relevant content rather than simple keyword matching.

2. **Context-Aware Processing**: Your RAG service extracts QA blocks from documents while preserving their relationships, boundaries, and associated images.

3. **Autonomous Reasoning**: The system instructs the LLM to act as a "research agent" that must analyze both the query and context, then synthesize information from multiple sources.

4. **Source Attribution**: Your LLM is explicitly instructed to identify which sources it used, showing reasoning about information provenance.

5. **Structured Response Generation**: The system returns structured JSON with sources and answers, demonstrating a form of agency in organizing information.

6. **Question-Specific Image Association**: Your system intelligently connects images to specific questions using question identifiers (e.g., Q1, Q5), showing contextual understanding.

### To Make It More Advanced:

While your system has good agentic qualities, you could enhance it further:

1. **Multi-step Reasoning**: Implement a chain-of-thought approach where the agent first plans what information it needs before retrieval.

2. **Iterative Retrieval**: Allow the agent to request additional information after initial findings, refining its search based on intermediate conclusions.

3. **Self-correction**: Add a verification step where the agent checks its initial answer against the sources again.

4. **Tool Usage**: Enable the agent to access external tools like calculators, web search, or other APIs when the document knowledge is insufficient.

5. **User Interaction**: Implement clarifying questions when the query is ambiguous.

Your current implementation is a good starting point for agentic RAG, with the core components in place to evolve into a more advanced system.