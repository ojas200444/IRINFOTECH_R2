import io
import pytest

def test_upload_document(client):
    # Prepare dummy file bytes
    file_content = b"%PDF-1.4 mock pdf data"
    file_bytes = io.BytesIO(file_content)

    # Post request to upload endpoint
    response = client.post(
        "/documents/upload",
        files={"files": ("test.pdf", file_bytes, "application/pdf")},
    )

    # Assert response
    assert response.status_code == 201
    data = response.json()
    assert "message" in data
    assert "documents" in data
    assert len(data["documents"]) == 1
    doc = data["documents"][0]
    assert doc["original_filename"] == "test.pdf"
    assert doc["status"] == "ready"
    assert doc["page_count"] == 1
    assert doc["chunk_count"] == 1

def test_upload_non_pdf(client):
    file_content = b"plain text data"
    file_bytes = io.BytesIO(file_content)

    response = client.post(
        "/documents/upload",
        files={"files": ("test.txt", file_bytes, "text/plain")},
    )

    assert response.status_code == 400
    assert "Only PDF files are accepted" in response.json()["message"]

def test_list_documents(client):
    response = client.get("/documents/")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total" in data

def test_get_document_details_and_delete(client):
    # 1. Upload first to ensure we have a document
    file_bytes = io.BytesIO(b"%PDF-1.4 mock pdf data")
    upload_res = client.post(
        "/documents/upload",
        files={"files": ("delete_me.pdf", file_bytes, "application/pdf")},
    )
    doc_id = upload_res.json()["documents"][0]["id"]

    # 2. Get document details
    response = client.get(f"/documents/{doc_id}")
    assert response.status_code == 200
    assert response.json()["original_filename"] == "delete_me.pdf"

    # 3. Delete document
    response = client.delete(f"/documents/{doc_id}")
    assert response.status_code == 204

    # 4. Try to get details again (should be 404)
    response = client.get(f"/documents/{doc_id}")
    assert response.status_code == 404
