from datetime import datetime
from pymongo.database import Database
from app.db.database import get_db
from typing import List, Dict

class ChatHistory:
    def __init__(self, db: Database = None):
        self.db = db or get_db()
        self.collection = self.db["chat_history"]

    def save_history(self, session_id: str, user: str, assistant: str):
        chat_entry = {
            "session_id": session_id,
            "timestamp": datetime.utcnow(),
            "user_message": user,
            "bot_response": assistant
        }
        result = self.collection.insert_one(chat_entry)
        return result.inserted_id

    def get_history(self, session_id: str, limit: int = 10) -> List[Dict[str, str]]:
        history_cursor = self.collection.find({"session_id": session_id}).sort("timestamp", -1).limit(limit)
        return [{"user_message": entry["user_message"], "bot_response": entry["bot_response"]} for entry in history_cursor]

    def delete_history(self, session_id: str) -> int:
        result = self.collection.delete_many({"session_id": session_id})
        return result.deleted_count

    def list_sessions(self) -> List[str]:
        return self.collection.distinct("session_id")
