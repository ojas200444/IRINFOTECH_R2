"""
app/services/__init__.py
========================
Services package for the RAG Document Assistant.

This package contains all business logic services:
- pdf_service:       PDF text extraction using PyMuPDF
- chunking_service:  Text chunking with recursive splitting
- embedding_service: Gemini embedding generation
- vector_store:      ChromaDB vector storage and retrieval
- rag_service:       RAG orchestration (embed → retrieve → generate)
- auth_service:      API key authentication
"""
