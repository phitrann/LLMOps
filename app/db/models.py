from pydantic import BaseModel
from datetime import datetime

class ChatMessage(BaseModel):
    session_id: str
    timestamp: datetime
    user_message: str
    bot_response: str
