# from fastapi import FastAPI
# from app.api import models

# app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the LLM API!"}

# @app.post("/infer")
# def infer(data: models.InferenceRequest):
#     # Gọi pipeline inference từ app.inference
#     from app.inference.pipeline import run_inference
#     result = run_inference(data.input_text)
#     return {"result": result}



import logging
from typing import AsyncGenerator, Literal

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from langchain.prompts import PromptTemplate
from langchain.schema import BaseChatMessageHistory, Document
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory

from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.chat_message_histories import FileChatMessageHistory

from app.inference.model import LLMModel, RawStreamer
from app.api.models import InferenceRequest, SSEResponse
from app.api.config import Settings
from app.api.utils import get_vectorstore, combine_documents, get_settings


# Create or import the FastAPI app instance
app = FastAPI(
    title="QA Chatbot Streaming using FastAPI, LangChain Expression Language , OpenAI, and Milvus",
    version="0.1.0",
)


async def generate_standalone_question(
    chat_history: str, question: str, settings: Settings
) -> str:
    prompt = PromptTemplate.from_template(
        template="""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
    )
    # llm = LLMModel(settings.MODEL_PATH, use_onnx=settings.USE_ONNX)
    llm = ChatOllama(model="phi3", base_url="http://172.16.87.75:11434")
    
    chain = prompt | llm | StrOutputParser()  # type: ignore

    return await chain.ainvoke(  # type: ignore
        {
            "chat_history": chat_history,
            "question": question,
        }
    )
    
async def search_relevant_documents(query: str, k: int = 5) -> list[Document]:
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever()

    return await retriever.aget_relevant_documents(query=query, k=k)

async def generate_response(
    context: str, chat_memory: BaseChatMessageHistory, message: str, settings: Settings
) -> AsyncGenerator[str, None]:
#     prompt = PromptTemplate.from_template(
#         """Answer the question based only on the following context:
# {context}
# Question: {question}"""
#     )

    prompt = PromptTemplate.from_template(
        """<|user|>\nAnswer the question based only on the following context:
{context} 
Question: {question} <|end|>\n<|assistant|>"""
    )

    llm = ChatOllama(model="phi3", base_url="http://172.16.87.75:11434")

    chain = prompt | llm  # type: ignore

    response = ""
    async for token in chain.astream({"context": context, "question": message}):  # type: ignore
        yield token.content
        response += token.content

    chat_memory.add_user_message(message=message)
    chat_memory.add_ai_message(message=response)

async def generate_sse_response(
    context: list[Document],
    chat_memory: BaseChatMessageHistory,
    message: str,
    settings: Settings,
) -> AsyncGenerator[str, SSEResponse]:
    prompt = PromptTemplate.from_template(
        """<|user|>\nAnswer the question based only on the following context:
{context} 
Question: {question} <|end|>\n<|assistant|>"""
    )

    llm = ChatOllama(model="phi3", base_url="http://172.16.87.75:11434")

    chain = prompt | llm  # type: ignore

    response = ""
    yield SSEResponse(type="context", value=context).json()
    try:
        yield SSEResponse(type="start", value="").json()
        async for token in chain.astream({"context": context, "question": message}):  # type: ignore
            yield SSEResponse(type="streaming", value=token.content).json()
            response += token.content

        yield SSEResponse(type="end", value="").json()
        chat_memory.add_user_message(message=message)
        chat_memory.add_ai_message(message=response)
    except Exception as e:  # TODO: Add proper exception handling
        yield SSEResponse(type="error", value=str(e)).json()

@app.post("/chat")
async def chat(
    request: InferenceRequest, settings: Settings = Depends(get_settings)
) -> StreamingResponse:
    memory_key = f"./message_store/{request.session_id}.json"

    chat_memory = FileChatMessageHistory(file_path=memory_key)
    memory = ConversationBufferMemory(chat_memory=chat_memory, return_messages=False)

    standalone_question = await generate_standalone_question(
        chat_history=memory.buffer, question=request.message, settings=settings
    )

    # relevant_documents = await search_relevant_documents(query=standalone_question)

    # combined_documents = combine_documents(relevant_documents)
    
    combined_documents = "You are an expert in all the fields. You can answer any question."

    return StreamingResponse(
        generate_response(
            context=combined_documents,
            chat_memory=chat_memory,
            message=request.message,
            settings=settings,
        ),
        media_type="text/plain",
    )
    
@app.post("/chat/sse/")
async def chat_sse(
    request: InferenceRequest, settings: Settings = Depends(get_settings)
) -> StreamingResponse:
    memory_key = f"./message_store/{request.session_id}.json"

    chat_memory = FileChatMessageHistory(file_path=memory_key)
    memory = ConversationBufferMemory(chat_memory=chat_memory, return_messages=False)

    standalone_question = await generate_standalone_question(
        chat_history=memory.buffer, question=request.message, settings=settings
    )

    relevant_documents = await search_relevant_documents(query=standalone_question, k=2)
    # relevant_documents = "You are an expert in all the fields. You can answer any question."

    return StreamingResponse(
        generate_sse_response(
            context=relevant_documents,
            chat_memory=chat_memory,
            message=request.message,
            settings=settings,
        ),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"}
    )