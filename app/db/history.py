import sqlite3
from contextlib import contextmanager
from langchain.schema import Document
from typing import List

DB_PATH = "chat_history.db"

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()

def create_table():
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                session_id TEXT,
                message TEXT,
                response TEXT
            )
            """
        )
        conn.commit()

def store_message(session_id: str, message: str, response: str):
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO chat_history (session_id, message, response) VALUES (?, ?, ?)",
            (session_id, message, response),
        )
        conn.commit()

def retrieve_history(session_id: str) -> List[Document]:
    with get_db_connection() as conn:
        cur = conn.execute(
            "SELECT message, response FROM chat_history WHERE session_id = ?",
            (session_id,),
        )
        rows = cur.fetchall()
    return [Document(page_content=f"User: {row[0]}\nAssistant: {row[1]}") for row in rows]

# Initialize the database
create_table()