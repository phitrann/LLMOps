from functools import lru_cache
from langchain.prompts import PromptTemplate
from langchain.schema import Document, format_document

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from app.api.config import Settings

@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore

@lru_cache()
def get_vectorstore() -> Milvus:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


    # The easiest way is to use Milvus Lite where everything is stored in a local file.
    # If you have a Milvus server you can use the server URI such as "http://localhost:19530".
    URI = "./milvus_example.db"

    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI},
    )

    return vector_store

async def search_relevant_documents(query: str, k: int = 5) -> list[Document]:
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return await retriever.ainvoke(input=query)

def combine_documents(
    docs: list[Document],
    document_prompt: PromptTemplate = PromptTemplate.from_template("{page_content}"),
    document_separator: str = "\n\n",
) -> str:
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

async def format_chat_history(history: list[dict]) -> str:
    return "\n".join([
        f"User: {entry['user_message']}\nAssistant: {entry['bot_response']}"
        for entry in history
    ])
    
def get_prompt(message: str, chat_history: list[dict[str, str]] = [], system_prompt: str = "You are a helpful AI assistant.") -> str:
    prompt = f"<|system|>\n{system_prompt}<|end|>\n"
    
    def get_chat_history(chat_history):
        texts = []
        for dialog in chat_history:
            texts.append(f"<|user|>\n{dialog['user_message'].strip()}<|end]>\n<|assistant|>\n{dialog['bot_response'].strip()}<|end|>\n")
        return "".join(texts)
    
    prompt += get_chat_history(chat_history)
    
    prompt += f"<|user|>\n{message.strip()}<|end|>\n<|assistant|>"
    return prompt