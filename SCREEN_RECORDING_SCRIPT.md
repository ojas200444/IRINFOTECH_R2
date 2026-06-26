# 🎬 Official Screen Recording Script: IR Infotech RAG Assistant

**Target Duration**: 15–20 minutes
**Objective**: Demonstrate the full capability of the Retrieval-Augmented Generation (RAG) API step-by-step.

---

## 🛑 Before You Start Recording:
1. Ensure your FastAPI server is **NOT** running yet.
2. Have a sample PDF ready on your desktop (e.g., your Biodata or a sample report).
3. Open two windows side-by-side or be ready to switch between them:
   - Your Code Editor (VS Code / Cursor) showing the terminal.
   - Your Web Browser (Chrome/Safari) with a blank tab.

---

## 🟢 1. Introduction (1-2 mins)

**(Start Screen Recording — Show your Code Editor)**

**🗣️ SPEAK THIS EXACTLY:**
> *"Hello! My name is Ojal, and this is my submission for Round 2 of the IR Infotech assignment: The RAG Application Development task.*
>
> *I have built a complete AI-powered Document Assistant from scratch. It is capable of extracting text from uploaded PDFs, generating vector embeddings using Google Gemini's embedding model, storing them in ChromaDB, and answering conversational questions based on the document context."*

**(Visual Action):** Briefly click through the folder structure in your editor (expand `app/routers`, `app/services`, etc.).

**🗣️ SPEAK THIS EXACTLY:**
> *"As you can see, the architecture is highly modular. We have separate services for PDF text extraction, document chunking, embedding generation, vector storage, and the final RAG LLM pipeline."*

---

## 🟢 2. Server Startup & Health Check (2-3 mins)

**(Visual Action):** Click inside your terminal. Type the command to start the server: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000` and press Enter.

**🗣️ SPEAK THIS EXACTLY:**
> *"Let's start the application server using Uvicorn. The backend is built on FastAPI, which provides incredibly fast performance and auto-generated API documentation."*

**(Visual Action):** Once the terminal says "Application startup complete", open your web browser and navigate to `http://localhost:8000/docs`.

**🗣️ SPEAK THIS EXACTLY:**
> *"I will navigate to localhost port 8000 slash docs. This opens the interactive Swagger UI where we can test all our endpoints directly.*
>
> *First, let's verify the system health. I will open the GET /health endpoint."*

**(Visual Action):** Click on `GET /health` to expand it. Click the **"Try it out"** button. Click the blue **"Execute"** button. Scroll down to show the JSON response.

**🗣️ SPEAK THIS EXACTLY:**
> *"As you can see in the response body, the API status is 'healthy', the database is connected, and our ChromaDB vector store is properly initialized and ready to receive data."*

---

## 🟢 3. Document Upload & Processing Pipeline (4-5 mins)

**(Visual Action):** Scroll down the Swagger UI page to the `POST /documents/upload` endpoint. Click to expand it. Click **"Try it out"**.

**🗣️ SPEAK THIS EXACTLY:**
> *"Now let's move to the ingestion pipeline. I will upload a PDF document. Behind the scenes, the application will use PyMuPDF to extract the raw text and page numbers. Then, it will pass that text to a Recursive Character Text Splitter to break it into overlapping chunks."*

**(Visual Action):** Click the **"Choose Files"** button. Select your sample PDF from your computer. Click the blue **"Execute"** button.

**🗣️ SPEAK THIS EXACTLY:**
> *"I am executing the upload now. Those text chunks are currently being sent to the Gemini Embedding API to be converted into vector embeddings. Finally, both the vectors and metadata are being saved into ChromaDB and SQLite."*

**(Visual Action):** Highlight the JSON response showing the success message, page count, and chunk count.

**🗣️ SPEAK THIS EXACTLY:**
> *"The upload is complete. The response shows exactly how many pages were parsed and how many chunks were generated and stored in the vector database."*

---

## 🟢 4. Question & Answering (RAG Pipeline) (5-6 mins)

**(Visual Action):** Scroll to the `POST /qa/ask` endpoint. Click to expand it. Click **"Try it out"**.

**🗣️ SPEAK THIS EXACTLY:**
> *"Next is the core feature: Question and Answering. When I ask a question, the system will embed my query, perform a semantic similarity search in ChromaDB, retrieve the top 5 most relevant chunks, and pass them as context to the Gemini 2.5 Flash LLM."*

**(Visual Action):** In the JSON Request body box, edit the text.
1. Change the `"question"` to something specific about your uploaded PDF. *(e.g., "What are the key skills listed in this document?")*
2. Under `"session_id"`, type `"test-session-1"`.
3. Click **"Execute"**.

**🗣️ SPEAK THIS EXACTLY:**
> *"I will ask the AI a specific question about the document and assign a session ID so we can track the conversation. Let's execute."*

**(Visual Action):** Wait a second for the response. Slowly scroll down to show the `Response body`. Highlight the `"answer"` text.

**🗣️ SPEAK THIS EXACTLY:**
> *"Here is the generated answer. As you can see, the LLM has successfully extracted the correct information. But more importantly, look right below the answer at the 'sources' array."*

**(Visual Action):** Highlight the `"sources"` array in the JSON response. Point out the `document_name`, `page_number`, and `chunk_text`.

**🗣️ SPEAK THIS EXACTLY:**
> *"To prevent hallucination, the API returns the exact source citations. We can see the original document name, the exact page number, the raw text snippet the AI used to generate the answer, and the vector relevance score. This ensures complete transparency."*

---

## 🟢 5. Chat History & Memory (3 mins)

**(Visual Action):** Scroll up slightly to the `GET /chat/sessions/{session_id}` endpoint. Click to expand it. Click **"Try it out"**.

**🗣️ SPEAK THIS EXACTLY:**
> *"The application also features conversational memory. All questions and answers are logged into the SQLite database. Let's retrieve the chat history for the session we just created."*

**(Visual Action):** In the `session_id` text box, type `"test-session-1"` (the exact same ID you used in the previous step). Click **"Execute"**.

**🗣️ SPEAK THIS EXACTLY:**
> *"When I execute this, you can see the complete chronology of the conversation. First, the user's question, followed by the assistant's response complete with all source citations. This history is automatically fed back into the LLM on subsequent questions so the AI remembers the context of our chat."*

---

## 🟢 6. Cleanup & Conclusion (1 min)

**(Visual Action):** Scroll to the `DELETE /documents/{document_id}` endpoint. Expand it, click **"Try it out"**. Type `1` in the `document_id` box. Click **"Execute"**.

**🗣️ SPEAK THIS EXACTLY:**
> *"Finally, the system supports full data lifecycle management. By calling the delete endpoint on our document ID, the system removes the metadata from SQLite, deletes the physical PDF from the server, and purges all associated vector embeddings from ChromaDB."*

**(Visual Action):** Show the `200` success response. Then switch back to your code editor/GitHub page.

**🗣️ SPEAK THIS EXACTLY:**
> *"The document has been successfully deleted. 
> That concludes the demonstration of my RAG Application Development submission. The complete source code, API documentation, and architecture diagrams are available on my GitHub repository. Thank you for your time!"*

**(End Screen Recording)**
