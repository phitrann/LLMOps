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

def combine_documents(
    docs: list[Document],
    document_prompt: PromptTemplate = PromptTemplate.from_template("{page_content}"),
    document_separator: str = "\n\n",
) -> str:
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)