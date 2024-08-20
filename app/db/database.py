from pymongo import MongoClient
from pymongo.database import Database

class MongoDB:
    def __init__(self, uri: str, db_name: str):
        self.client = MongoClient(uri)
        self.db_name = db_name

    def get_database(self) -> Database:
        return self.client[self.db_name]

# Singleton instance to be used across the app
mongo_instance = MongoDB(uri="mongodb://localhost:27017/", db_name="chatbot_db")

def get_db() -> Database:
    return mongo_instance.get_database()
