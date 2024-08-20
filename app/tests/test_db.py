import pytest
import mongomock
from app.db.database import MongoDB, get_db
from app.db.history import ChatHistory

chat_history = ChatHistory()

# Mock MongoDB URI and Database name for testing
TEST_MONGO_URI = "mongodb://localhost:27017/"
TEST_DB_NAME = "test_chatbot_db"

@pytest.fixture(scope="module")
def mock_db():
    """Fixture to create a mock MongoDB instance."""
    mock_client = mongomock.MongoClient(TEST_MONGO_URI)
    mock_db = mock_client[TEST_DB_NAME]
    return mock_db

@pytest.fixture(scope="module")
def patch_get_db(mock_db):
    """Fixture to patch get_db function to return the mock database."""
    original_get_db = get_db

    def mock_get_db():
        return mock_db

    # Patch the function
    MongoDB.get_database = mock_get_db
    yield
    # Revert to original function
    MongoDB.get_database = original_get_db

def test_get_db(mock_db, patch_get_db):
    """Test that get_db returns the correct mock database."""
    db = get_db()
    assert db.name == TEST_DB_NAME

def test_save_chat_history(mock_db, patch_get_db):
    """Test saving chat history entry."""
    session_id = "test_session_1"
    user_message = "Hello"
    bot_response = "Hi there!"

    chat_history.save_history(session_id, user_message, bot_response)

    saved_history = list(mock_db["chat_history"].find({"session_id": session_id}))
    assert len(saved_history) == 1
    assert saved_history[0]["user_message"] == user_message
    assert saved_history[0]["bot_response"] == bot_response

def test_save_chat_history_duplicate(mock_db, patch_get_db):
    """Test saving duplicate chat history entry."""
    session_id = "test_session_2"
    user_message = "How are you?"
    bot_response = "I'm good, thanks!"

    # Save first entry
    chat_history.save_history(session_id, user_message, bot_response)

    # Attempt to save duplicate
    chat_history.save_history(session_id, user_message, bot_response)

    saved_history = list(mock_db["chat_history"].find({"session_id": session_id}))
    assert len(saved_history) == 2

def test_get_chat_history(mock_db, patch_get_db):
    """Test retrieving chat history."""
    session_id = "test_session_3"
    user_message_1 = "Hi"
    bot_response_1 = "Hello!"
    user_message_2 = "What's your name?"
    bot_response_2 = "I'm a chatbot."

    mock_db["chat_history"].insert_many([
        {"session_id": session_id, "user_message": user_message_1, "bot_response": bot_response_1, "timestamp": 1},
        {"session_id": session_id, "user_message": user_message_2, "bot_response": bot_response_2, "timestamp": 2}
    ])

    history = chat_history.get_history(session_id)
    assert len(history) == 2
    assert history[0]["user_message"] == user_message_1
    assert history[0]["bot_response"] == bot_response_1
    assert history[1]["user_message"] == user_message_2
    assert history[1]["bot_response"] == bot_response_2

def test_get_chat_history_empty(mock_db, patch_get_db):
    """Test retrieving chat history for a session with no history."""
    session_id = "non_existent_session"
    history = chat_history.get_history(session_id)
    assert history == []

def test_delete_chat_history(mock_db, patch_get_db):
    """Test deleting chat history for a session."""
    session_id = "test_session_4"
    user_message = "Will this be deleted?"
    bot_response = "Yes, it will."

    chat_history.save_history(session_id, user_message, bot_response)
    deleted_count = chat_history.delete_history(session_id)

    assert deleted_count == 1
    history = chat_history.get_history(session_id)
    assert history == []

def test_list_sessions(mock_db, patch_get_db):
    """Test listing all session IDs."""
    session_id_1 = "session_1"
    session_id_2 = "session_2"

    mock_db["chat_history"].insert_many([
        {"session_id": session_id_1, "user_message": "Hello", "bot_response": "Hi", "timestamp": 1},
        {"session_id": session_id_2, "user_message": "How are you?", "bot_response": "Good", "timestamp": 2}
    ])

    sessions = chat_history.list_sessions()
    assert set(sessions) == {session_id_1, session_id_2}

def test_save_chat_history_error_handling(mock_db, patch_get_db):
    """Test error handling when trying to save invalid data."""
    session_id = "test_session_5"
    user_message = {"invalid": "data"}  # Invalid data type
    bot_response = "This should fail"

    with pytest.raises(TypeError):
        chat_history.save_history(session_id, user_message, bot_response)
