"""
app/routers/documents.py
=========================
Defines all document management API endpoints.

This router handles the full document lifecycle:
1. Upload — Accept PDF files, extract text, chunk, embed, and store
2. List   — View all uploaded documents
3. Get    — View details of a single document
4. Delete — Remove a document from DB, vector store, and filesystem
"""

import os
import uuid
import aiofiles
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models.db_models import Document
from app.models.response_models import (
    UploadResponse,
    DocumentResponse,
    DocumentListResponse,
    ErrorResponse,
)
from app.services.pdf_service import extract_text_from_bytes, count_pages_from_bytes
from app.services.chunking_service import chunk_text
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.auth_service import get_api_key_dependency
from app.logger import setup_logger

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

logger = setup_logger(__name__)
settings = get_settings()

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
    dependencies=[Depends(get_api_key_dependency)],
)


# ─────────────────────────────────────────────
# HELPER — Save uploaded file to disk
# ─────────────────────────────────────────────

async def _save_upload(file: UploadFile, upload_dir: str) -> tuple[str, bytes]:
    """
    Save an uploaded file to the uploads directory and return
    both the saved file path and the raw file bytes.
    """
    unique_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = os.path.join(upload_dir, unique_name)

    # Read the entire file into memory
    file_bytes = await file.read()

    # Write to disk asynchronously
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(file_bytes)

    logger.info(f"Saved uploaded file to: {file_path} ({len(file_bytes)} bytes)")
    return file_path, file_bytes


# ─────────────────────────────────────────────
# POST /documents/upload — Upload & process PDFs
# ─────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload PDF Documents",
    description=(
        "Upload one or more PDF files for processing. Each file goes through "
        "the full RAG ingestion pipeline: text extraction → chunking → "
        "embedding → vector storage."
    ),
    responses={
        201: {"description": "Documents uploaded and processed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid file type (only PDFs allowed)"},
        422: {"description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Processing error"},
    },
)
async def upload_documents(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> UploadResponse:
    """
    ## Upload PDF Documents
    """
    files = [file]
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided. Please upload at least one PDF file.",
        )

    logger.info(f"POST /documents/upload — Received {len(files)} file(s)")

    upload_dir = settings.upload_dir
    os.makedirs(upload_dir, exist_ok=True)

    embedding_service = EmbeddingService()
    vector_store = VectorStoreService()

    processed_documents: List[DocumentResponse] = []
    failed_count = 0

    for file in files:
        try:
            # Validate file type
            if not file.filename or not file.filename.lower().endswith(".pdf"):
                logger.warning(f"Rejected non-PDF file: {file.filename}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Only PDF files are accepted. Got: '{file.filename}'",
                )

            logger.info(f"Processing file: {file.filename}")

            # Step 1: Save to disk
            file_path, file_bytes = await _save_upload(file, upload_dir)

            # Step 2: Get page count and extract text
            page_count = count_pages_from_bytes(file_bytes)
            extracted_text = extract_text_from_bytes(file_bytes)
            if not extracted_text or not any(page.get("text", "").strip() for page in extracted_text):
                logger.warning(f"No text extracted from: {file.filename}")
                raise ValueError(f"Could not extract any text from '{file.filename}'. The PDF may be image-based or empty.")

            logger.info(
                f"Extracted pages from {file.filename}"
            )

            # Step 3: Chunk the text
            chunks = chunk_text(extracted_text)
            logger.info(f"Created {len(chunks)} chunks from {file.filename}")

            # Step 4: Generate embeddings (synchronous service call)
            embeddings = embedding_service.generate_embeddings(
                [chunk["text"] for chunk in chunks]
            )
            logger.info(f"Generated {len(embeddings)} embeddings for {file.filename}")

            # Step 5: Create DB record (synchronous)
            document = Document(
                filename=os.path.basename(file_path),
                original_filename=file.filename,
                file_size=len(file_bytes),
                page_count=page_count,
                chunk_count=len(chunks),
                status="ready",
            )
            db.add(document)
            db.flush()  # Flush to get the auto-generated ID

            # Step 6: Store in ChromaDB (synchronous)
            vector_store.add_document(
                doc_id=document.id,
                chunks=chunks,
                embeddings=embeddings,
                document_name=document.original_filename,
            )
            logger.info(
                f"Stored {len(chunks)} vectors in ChromaDB for document {document.id}"
            )

            # Step 7: Commit transaction
            db.commit()
            db.refresh(document)

            # Build response model
            processed_documents.append(
                DocumentResponse(
                    id=document.id,
                    filename=document.filename,
                    original_filename=document.original_filename,
                    file_size=document.file_size,
                    page_count=document.page_count,
                    chunk_count=document.chunk_count,
                    upload_date=document.upload_date.isoformat(),
                    status=document.status,
                )
            )

            logger.info(
                f"✅ Successfully processed: {file.filename} → "
                f"Document ID: {document.id}, Chunks: {len(chunks)}"
            )

        except HTTPException:
            raise

        except Exception as e:
            failed_count += 1
            logger.error(
                f"❌ Failed to process {file.filename}: {str(e)}",
                exc_info=True,
            )
            db.rollback()

    if failed_count > 0 and len(processed_documents) == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="All files failed to process. Check the server logs for details.",
        )

    logger.info(
        f"Upload complete — {len(processed_documents)} succeeded, {failed_count} failed"
    )

    return UploadResponse(
        message=f"Successfully processed {len(processed_documents)} document(s)",
        documents=processed_documents,
    )


# ─────────────────────────────────────────────
# GET /documents — List all documents
# ─────────────────────────────────────────────

@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List All Documents",
    description="Retrieve a list of all uploaded documents with their metadata.",
    responses={
        200: {"description": "List of documents returned successfully"},
        500: {"model": ErrorResponse, "description": "Database error"},
    },
)
async def list_documents(
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    """
    ## List All Documents
    """
    logger.info("GET /documents — Listing all documents")

    try:
        documents = db.query(Document).order_by(Document.upload_date.desc()).all()

        document_list = [
            DocumentResponse(
                id=doc.id,
                filename=doc.filename,
                original_filename=doc.original_filename,
                file_size=doc.file_size,
                page_count=doc.page_count,
                chunk_count=doc.chunk_count,
                upload_date=doc.upload_date.isoformat(),
                status=doc.status,
            )
            for doc in documents
        ]

        logger.info(f"Returning {len(document_list)} document(s)")

        return DocumentListResponse(
            documents=document_list,
            total=len(document_list),
        )

    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve documents: {str(e)}",
        )


# ─────────────────────────────────────────────
# GET /documents/{document_id} — Get single document
# ─────────────────────────────────────────────

@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get Document Details",
    description="Retrieve detailed information about a specific document by its ID.",
    responses={
        200: {"description": "Document details returned successfully"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Database error"},
    },
)
async def get_document(
    document_id: int,
    db: Session = Depends(get_db),
) -> DocumentResponse:
    """
    ## Get Document Details
    """
    logger.info(f"GET /documents/{document_id} — Fetching document details")

    try:
        document = db.get(Document, document_id)

        if document is None:
            logger.warning(f"Document not found: {document_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found.",
            )

        return DocumentResponse(
            id=document.id,
            filename=document.filename,
            original_filename=document.original_filename,
            file_size=document.file_size,
            page_count=document.page_count,
            chunk_count=document.chunk_count,
            upload_date=document.upload_date.isoformat(),
            status=document.status,
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            f"Failed to fetch document {document_id}: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document: {str(e)}",
        )


# ─────────────────────────────────────────────
# DELETE /documents/{document_id} — Delete a document
# ─────────────────────────────────────────────

@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Document",
    description=(
        "Delete a document and all its associated data from the database, "
        "vector store, and filesystem."
    ),
    responses={
        204: {"description": "Document deleted successfully"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Deletion error"},
    },
)
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
) -> None:
    """
    ## Delete Document
    """
    logger.info(f"DELETE /documents/{document_id} — Deleting document")

    try:
        document = db.get(Document, document_id)

        if document is None:
            logger.warning(f"Document not found for deletion: {document_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID {document_id} not found.",
            )

        # Step 1: Delete from ChromaDB vector store (passes int, synchronous)
        try:
            vector_store = VectorStoreService()
            vector_store.delete_document(document.id)
            logger.info(f"Deleted vectors for document {document_id} from ChromaDB")
        except Exception as e:
            logger.warning(
                f"Failed to delete vectors for document {document_id}: {str(e)}"
            )

        # Step 2: Delete PDF file from disk
        # We need to construct the full file path.
        # Since file_path isn't a column anymore in db_models, we find it from filename
        upload_dir = settings.upload_dir
        file_path = os.path.join(upload_dir, document.filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
            except OSError as e:
                logger.warning(
                    f"Failed to delete file {file_path}: {str(e)}"
                )

        # Step 3: Delete from database
        db.delete(document)
        db.commit()

        logger.info(f"✅ Successfully deleted document {document_id}")
        return None

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            f"Failed to delete document {document_id}: {str(e)}", exc_info=True
        )
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}",
        )
