import pytest

def test_ask_question_standard(client):
    payload = {
        "question": "What is the key conclusion of the report?",
        "stream": False
    }

    response = client.post("/qa/ask", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "session_id" in data
    assert len(data["sources"]) > 0
    assert data["sources"][0]["document_name"] == "sample.pdf"

def test_ask_question_streaming(client):
    payload = {
        "question": "Can you summarize this document?",
        "stream": True
    }

    response = client.post("/qa/ask", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    # Read SSE events from streaming response
    lines = response.content.decode("utf-8").split("\n\n")
    events = [line for line in lines if line.strip()]
    
    assert len(events) > 0
    # First chunk event should have "token"
    assert "token" in events[0]
    # One of the events should indicate "done" and have sources
    assert any("done" in event and "sources" in event for event in events)
