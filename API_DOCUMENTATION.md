# IR Infotech RAG Assistant — API Documentation

This backend is built with FastAPI. Interactive documentation is automatically available at `http://localhost:8000/docs` (Swagger UI) when the server is running.

## Base URL
`http://localhost:8000`

---

### 1. Health & Status

#### `GET /health`
Checks the health of the API, vector store, and SQLite database.
- **Response**: `200 OK`
```json
{
  "status": "healthy",
  "app": "RAG Assistant API",
  "version": "1.0.0",
  "documents_count": 1,
  "vector_store_status": "connected"
}
```

---

### 2. Document Management

#### `POST /documents/upload`
Upload one or more PDF documents. The system extracts text, chunks it, generates vector embeddings via Gemini, and stores them in ChromaDB.
- **Request Body**: `multipart/form-data` with one or more `files` (PDFs).
- **Response**: `200 OK`
```json
{
  "message": "Successfully uploaded and processed 1 file(s).",
  "documents": [
    {
      "id": 1,
      "filename": "uuid_Document.pdf",
      "original_filename": "Document.pdf",
      "file_size": 102400,
      "page_count": 10,
      "chunk_count": 45,
      "upload_date": "2026-06-25T12:00:00Z",
      "status": "ready"
    }
  ]
}
```

#### `GET /documents`
List all uploaded documents and their processing status.
- **Response**: `200 OK`
```json
{
  "documents": [...],
  "total": 1
}
```

#### `DELETE /documents/{doc_id}`
Delete a document and remove its embeddings from the vector store.
- **Response**: `200 OK`
```json
{
  "message": "Document 1 deleted successfully"
}
```

---

### 3. Question Answering (RAG)

#### `POST /qa/ask`
Ask a question based on uploaded documents. Retrieves relevant context from the vector database and generates an AI answer.
- **Request Body**:
```json
{
  "question": "What is the summary of the report?",
  "document_ids": [1],           // Optional: Restrict to specific documents
  "session_id": "abc-123",       // Optional: Group messages into a chat thread
  "stream": false,               // Set true for Server-Sent Events (SSE) streaming
  "top_k": 5                     // Number of relevant chunks to retrieve
}
```
- **Response**: `200 OK`
```json
{
  "answer": "The report summarizes...",
  "sources": [
    {
      "document_id": 1,
      "document_name": "Document.pdf",
      "page_number": 2,
      "chunk_text": "...",
      "relevance_score": 0.89
    }
  ],
  "session_id": "abc-123",
  "question": "What is the summary of the report?"
}
```

---

### 4. Chat History

#### `GET /chat/sessions`
Retrieve a list of all chat sessions.
- **Response**: `200 OK`
```json
[
  {
    "id": "abc-123",
    "title": "New Chat",
    "created_at": "...",
    "updated_at": "...",
    "message_count": 2
  }
]
```

#### `GET /chat/sessions/{session_id}`
Retrieve the full message history (including AI responses and citations) for a specific session.
- **Response**: `200 OK`
```json
{
  "session_id": "abc-123",
  "title": "New Chat",
  "messages": [
    {
      "id": 1,
      "role": "user",
      "content": "What is the summary?",
      "sources": null,
      "created_at": "..."
    },
    {
      "id": 2,
      "role": "assistant",
      "content": "The report summarizes...",
      "sources": [...],
      "created_at": "..."
    }
  ]
}
```

#### `DELETE /chat/sessions/{session_id}`
Delete a chat session and all its messages.
- **Response**: `200 OK`
