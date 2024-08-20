from pymongo.database import Database
from app.db.database import get_db

class VectorDB:
    def __init__(self, db: Database = None):
        self.db = db or get_db()
        self.collection = self.db["vectors"]

    def store_vector(self, vector_id: str, vector_data: list):
        vector_entry = {
            "vector_id": vector_id,
            "vector_data": vector_data
        }
        self.collection.insert_one(vector_entry)

    def get_vector(self, vector_id: str):
        return self.collection.find_one({"vector_id": vector_id})

    def delete_vector(self, vector_id: str):
        result = self.collection.delete_one({"vector_id": vector_id})
        return result.deleted_count

    def search_vectors(self, query_vector: list, top_k: int = 10):
        # Implement your vector search logic here
        pass
