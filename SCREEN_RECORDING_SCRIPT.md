# Screen Recording Script (IR Infotech R2 RAG Assistant)

**Target Duration**: 15–20 minutes
**Objective**: Demonstrate the full capability of the Retrieval-Augmented Generation (RAG) API, including file uploads, text extraction, embeddings, and context-aware Q&A using Swagger UI.

---

## 1. Introduction (1 min)
* **Action**: Start recording with your screen showing the code editor (VS Code/Cursor) and terminal.
* **Script**: "Hello! This is my submission for Round 2 — the RAG Application Development task for IR Infotech. I have built an AI-powered Document Assistant capable of answering questions directly from uploaded PDFs using FastAPI, ChromaDB, SQLite, and Google Gemini."
* **Action**: Briefly show the project structure (the `app/` folder, `routers/`, `services/`, and `.env`). 
* **Script**: "The application handles text extraction, chunking, embedding generation using Gemini's embedding model, and vector storage."

## 2. Server Startup & Health Check (2 mins)
* **Action**: In the terminal, run `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`. Wait for it to say the application startup is complete.
* **Action**: Open your browser to `http://localhost:8000/docs`.
* **Script**: "I am starting the FastAPI server. We can interact with all the endpoints through this auto-generated Swagger UI."
* **Action**: Open the `GET /health` endpoint, click "Try it out", then "Execute".
* **Script**: "First, let's hit the health check endpoint. As you can see, the API is healthy, and the ChromaDB vector store is properly connected."

## 3. Document Upload & Processing (4 mins)
* **Action**: Scroll down to `POST /documents/upload`. Click "Try it out".
* **Action**: Click "Choose Files" and select 1 or 2 PDF files (e.g., your Biodata or any sample PDF).
* **Action**: Click "Execute" and wait for the response.
* **Script**: "Now I will upload a PDF document. When I hit execute, the backend uses PyMuPDF to extract the text, splits the text into chunks with some overlap, calls the Gemini Embedding API to turn those chunks into vectors, and saves them into ChromaDB. It also saves the metadata to SQLite."
* **Action**: Show the `200` response body indicating the file was successfully uploaded, along with the chunk count.
* **Action**: Call the `GET /documents` endpoint to show the document is listed in the database.

## 4. Question & Answering (RAG) (6 mins)
* **Action**: Go to the `POST /qa/ask` endpoint. Click "Try it out".
* **Action**: In the request body, type a specific question related to the PDF you just uploaded. 
  *(Example: `"question": "What are Ojal's key technical skills?"`)*
* **Action**: Set a random `session_id` (e.g., `"session_id": "test-session-123"`).
* **Action**: Click "Execute".
* **Script**: "Now for the main feature. I am asking a question based on the document. The system embeds my question, searches ChromaDB for the most relevant chunks, and passes them as context to the Gemini 2.5 Flash LLM."
* **Action**: Highlight the answer in the response body. 
* **Script**: "Here is the AI's generated answer. Notice the `sources` array right below it! It provides the exact document name, page number, and the raw text chunk it used to generate this answer, along with a relevance score. This ensures the AI is fully grounded in the documents and prevents hallucination."

## 5. Chat History & Memory (4 mins)
* **Action**: Go to `GET /chat/sessions`. Execute it.
* **Script**: "The application also tracks conversational history. If I check the sessions endpoint, you can see the session ID I just used."
* **Action**: Copy the `session_id`. Go to `GET /chat/sessions/{session_id}`. Paste the ID and execute.
* **Script**: "When I fetch the history for this session, you can see the complete history: my original question as the 'user', and the generated answer (with sources) as the 'assistant'. This history is passed back to the LLM on subsequent questions so it remembers the context of our conversation."

## 6. Cleanup & Conclusion (2 mins)
* **Action**: Go to `DELETE /documents/{doc_id}`. Enter the ID of the document you uploaded (e.g., `1`) and execute.
* **Script**: "Finally, we can delete the document. This removes the metadata from SQLite, the physical file from the uploads folder, and deletes the vector collection from ChromaDB."
* **Action**: Show the `200` success response.
* **Script**: "This concludes the demonstration. The architecture diagram, API documentation, and source code are available in the GitHub repository. Thank you!"
