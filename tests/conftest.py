import os
import pytest
from unittest.mock import MagicMock, patch

# Set environment variables for testing
os.environ["DATABASE_URL"] = "sqlite:///./test_rag_assistant.db"
os.environ["GEMINI_API_KEY"] = "fake-key"
os.environ["API_KEY_AUTH"] = "test-api-key"

# Make sure we import app after setting environment variables
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app

# Test engine
engine = create_engine(
    os.environ["DATABASE_URL"],
    connect_args={"check_same_thread": False},
)
TestingSessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    # Create test database tables
    Base.metadata.create_all(bind=engine)
    yield
    # Clean up test database
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("./test_rag_assistant.db"):
        try:
            os.remove("./test_rag_assistant.db")
        except OSError:
            pass

@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture(autouse=True)
def override_get_db(db_session):
    app.dependency_overrides[get_db] = lambda: db_session
    yield
    app.dependency_overrides.clear()

@pytest.fixture
def client():
    with TestClient(app) as c:
        c.headers["X-API-Key"] = "test-api-key"
        yield c

# Mock embedding service and vector store to avoid actual API calls
@pytest.fixture(autouse=True)
def mock_external_services():
    with patch("app.routers.documents.EmbeddingService") as mock_emb_doc, \
         patch("app.routers.documents.VectorStoreService") as mock_vs_doc, \
         patch("app.routers.qa.RAGService") as mock_rag_qa, \
         patch("app.routers.documents.count_pages_from_bytes", return_value=1), \
         patch("app.routers.documents.extract_text_from_bytes", return_value=[{"page_number": 1, "text": "Sample document text"}]):
        
        # Setup mocks
        emb_instance = mock_emb_doc.return_value
        emb_instance.generate_embeddings.return_value = [[0.1] * 768]
        emb_instance.generate_embedding.return_value = [0.1] * 768

        vs_instance = mock_vs_doc.return_value
        vs_instance.add_document.return_value = None
        vs_instance.delete_document.return_value = None
        vs_instance.get_collection_count.return_value = 0

        rag_instance = mock_rag_qa.return_value

        async def mock_ask(*args, **kwargs):
            return {
                "answer": "This is a mock answer based on the document context.",
                "sources": [
                    {
                        "document_id": 1,
                        "document_name": "sample.pdf",
                        "page_number": 1,
                        "chunk_text": "Sample text chunk from document.",
                        "relevance_score": 0.95
                    }
                ]
            }
        rag_instance.ask = mock_ask

        
        async def mock_ask_stream(*args, **kwargs):
            yield {"type": "chunk", "content": "This is "}
            yield {"type": "chunk", "content": "a mock streamed answer."}
            yield {
                "type": "sources",
                "content": [
                    {
                        "document_id": 1,
                        "document_name": "sample.pdf",
                        "page_number": 1,
                        "chunk_text": "Sample text chunk.",
                        "relevance_score": 0.95
                    }
                ]
            }
            yield {"type": "done", "content": ""}
            
        rag_instance.ask_stream = mock_ask_stream

        yield {
            "embedding": emb_instance,
            "vector_store": vs_instance,
            "rag": rag_instance
        }
