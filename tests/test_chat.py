import pytest

def test_chat_sessions_flow(client):
    # 1. Ask a question to create a session
    payload = {
        "question": "What is in the PDF?",
        "stream": False
    }
    qa_response = client.post("/qa/ask", json=payload)
    assert qa_response.status_code == 200
    session_id = qa_response.json()["session_id"]
    assert session_id is not None

    # 2. List chat sessions
    sessions_response = client.get("/chat/sessions")
    assert sessions_response.status_code == 200
    sessions = sessions_response.json()
    assert len(sessions) > 0
    # Find our session in the list
    session_ids = [s["id"] for s in sessions]
    assert session_id in session_ids

    # 3. Get history for this session
    history_response = client.get(f"/chat/sessions/{session_id}")
    assert history_response.status_code == 200
    history_data = history_response.json()
    assert history_data["session_id"] == session_id
    assert len(history_data["messages"]) == 2 # 1 user question + 1 assistant answer
    assert history_data["messages"][0]["role"] == "user"
    assert history_data["messages"][1]["role"] == "assistant"

    # 4. Delete the session
    delete_response = client.delete(f"/chat/sessions/{session_id}")
    assert delete_response.status_code == 204

    # 5. Verify the session is gone (should return 404)
    history_response_after = client.get(f"/chat/sessions/{session_id}")
    assert history_response_after.status_code == 404
