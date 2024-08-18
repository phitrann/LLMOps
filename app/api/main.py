import logging
from typing import AsyncGenerator, Literal
from threading import Lock
import asyncio

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from langchain.prompts import PromptTemplate
from langchain.schema import BaseChatMessageHistory, Document
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory

from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.chat_message_histories import FileChatMessageHistory

from app.inference.model import LLMModel
from app.api.models import InferenceRequest, SSEResponse
from app.api.config import Settings
from app.api.utils import search_relevant_documents, combine_documents, get_settings


# Create or import the FastAPI app instance
app = FastAPI(
    title="QA Chatbot Streaming using FastAPI, LangChain Expression Language , OpenAI, and Milvus",
    version="0.1.0",
)

@app.on_event("startup")
async def startup_event():
    global settings 
    settings = get_settings()
    
    
    global llm
    
    logging.info("Loadding LLM model...")
    # llm = ChatOllama(model="phi3:instruct", base_url="http://172.16.87.75:11434")
    llm = LLMModel(settings)
    logging.info("LLM model loaded successfully.")
    
llama_outer_lock = Lock()
llama_inner_lock = Lock()

def get_llama():
    # NOTE: This double lock allows the currently streaming llama model to
    # check if any other requests are pending in the same thread and cancel
    # the stream if so.
    llama_outer_lock.acquire()
    release_outer_lock = True
    try:
        llama_inner_lock.acquire()
        try:
            llama_outer_lock.release()
            release_outer_lock = False
            yield llm
        finally:
            llama_inner_lock.release()
    finally:
        if release_outer_lock:
            llama_outer_lock.release()
    
    
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
    
    # chain = prompt | llm | StrOutputParser()  # type: ignore

    # return await chain.ainvoke(  # type: ignore
    #     {
    #         "chat_history": chat_history,
    #         "question": question,
    #     }
    # )
    result = await llm(prompt, chat_history=chat_history, question=question)
    return str(result)
    
    

from threading import Thread
async def generate_response(
    context: str, chat_memory: BaseChatMessageHistory, message: str, settings: Settings
) -> AsyncGenerator[str, None]:

    prompt = PromptTemplate.from_template(
        """<|user|>\nAnswer the question based only on the following context:
{context} 
Question: {question} <|end|>\n<|assistant|>"""
    )

    # chain = prompt | llm  # type: ignore

    # response = ""
    # async for token in chain.astream({"context": context, "question": message}):  # type: ignore
    #     yield token.content
    #     response += token.content
    
    response_chunks = llm.generate(prompt, context=context, question=message)
    response = ""
    
    async for chunk in response_chunks:
        response += chunk
        yield chunk

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

    # chain = prompt | llm  # type: ignore

    response = ""
    yield SSEResponse(type="context", value=context).json()
    try:
        yield SSEResponse(type="start", value="").json()
        # async for token in chain.astream({"context": context, "question": message}):  # type: ignore
        #     yield SSEResponse(type="streaming", value=token.content).json()
        #     response += token.content
        #     print(response)
        async for token in llm.generate(prompt, context=context, question=message):
            yield SSEResponse(type="streaming", value=token).json()
            response += token
            print(response)

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
    logging.info(f"Received message: {request.message}")

    chat_memory = FileChatMessageHistory(file_path=memory_key)
    memory = ConversationBufferMemory(chat_memory=chat_memory, return_messages=False)

    logging.info(f"Generating standalone question for: {request.message}")
    # standalone_question = await generate_standalone_question(
    #     chat_history=memory.buffer, question=request.message, settings=settings
    # )
    
    # logging.info(f"Searching for relevant documents for: {standalone_question}")

    # relevant_documents = await search_relevant_documents(query=standalone_question)

    # combined_documents = combine_documents(relevant_documents)
    
    combined_documents = "You are an expert in all the fields. You can answer any question."

    logging.info(f"Generating response for: {request.message}")

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

    # standalone_question = await generate_standalone_question(
    #     chat_history=memory.buffer, question=request.message, settings=settings
    # )

    # relevant_documents = await search_relevant_documents(query=standalone_question, k=2)
    relevant_documents = "You are an expert in all the fields. You can answer any question."

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