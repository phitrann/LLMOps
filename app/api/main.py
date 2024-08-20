"""
https://gist.github.com/jvelezmagic/f3653cc2ddab1c91e86751c8b423a1b6
"""

import logging
from typing import AsyncGenerator, Literal
from threading import Lock
import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from langchain.prompts import PromptTemplate
from langchain.schema import BaseChatMessageHistory, Document
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory

from app.inference.model import LLMModel
from app.api.models import InferenceRequest, SSEResponse
from app.api.config import Settings
from app.api.utils import (
    search_relevant_documents,
    combine_documents,
    get_settings,
    get_prompt,
)
from app.db.history import ChatHistory


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):  # https://fastapi.tiangolo.com/fa/advanced/events/#lifespan
    global settings
    settings = get_settings()

    global llm, chat_history

    logging.info("Loadding LLM model...")
    llm = LLMModel(settings)
    logging.info("LLM model loaded successfully.")

    chat_history = ChatHistory()

    yield

    del llm
    del settings


# Create or import the FastAPI app instance
app = FastAPI(
    title="QA Chatbot Streaming using FastAPI, LangChain Expression Language , OpenAI, and Milvus",
    version="0.1.0",
    lifespan=lifespan,
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

    result = await llm(prompt, chat_history=chat_history, question=question)
    return str(result)


from queue import Empty

# Using run_in_threadpool
from starlette.concurrency import run_in_threadpool


async def generate_response(
    context: str,
    chat_memory: list[dict[str, str]],
    message: str,
    session_id: str,
    settings: Settings,
) -> AsyncGenerator[str, None]:

    prompt = get_prompt(message=message, chat_history=chat_memory)

    response = ""
    streamer = llm.generate(prompt, context=context, question=message)

    async for token in streamer:
        response += token
        yield token
        await asyncio.sleep(0.001)

    chat_history.save_history(session_id=session_id, user=message, assistant=response)


async def generate_sse_response(
    context: list[Document],
    chat_memory: list[dict[str, str]],
    message: str,
    session_id: str,
    settings: Settings,
) -> AsyncGenerator[str, SSEResponse]:
    prompt = get_prompt(message=message, chat_history=chat_memory)

    # chain = prompt | llm  # type: ignore

    response = ""
    yield SSEResponse(type="context", value=context).json()
    try:
        yield SSEResponse(type="start", value="").json()
        async for token in llm.generate(prompt, context=context, question=message):
            yield SSEResponse(type="streaming", value=token).json()
            response += token
            await asyncio.sleep(0.001)
            # print(token)

        yield SSEResponse(type="end", value="").json()
        chat_history.save_history(
            session_id=session_id, user=message, assistant=response
        )
    except Exception as e:  # TODO: Add proper exception handling
        yield SSEResponse(type="error", value=str(e)).json()


@app.post("/chat")
async def chat(
    request: InferenceRequest, settings: Settings = Depends(get_settings)
) -> StreamingResponse:
    logging.info(f"Received message: {request.message}")

    # logging.info(f"Searching for relevant documents for: {standalone_question}")

    # relevant_documents = await search_relevant_documents(query=standalone_question)

    # combined_documents = combine_documents(relevant_documents)

    combined_documents = (
        ""
    )

    logging.info(f"Generating response for: {request.message}")

    return StreamingResponse(
        generate_response(
            context=combined_documents,
            chat_memory=chat_history.get_history(request.session_id, limit=5),
            message=request.message,
            session_id=request.session_id,
            settings=settings,
        ),
        media_type="text/plain",
    )


@app.post("/chat/sse/")
async def chat_sse(
    request: InferenceRequest, settings: Settings = Depends(get_settings)
) -> StreamingResponse:

    # standalone_question = await generate_standalone_question(
    #     chat_history=memory.buffer, question=request.message, settings=settings
    # )

    # relevant_documents = await search_relevant_documents(query=standalone_question, k=2)
    relevant_documents = (
        ""
    )

    return StreamingResponse(
        generate_sse_response(
            context=relevant_documents,
            chat_memory=chat_history.get_history(request.session_id, limit=5),
            message=request.message,
            session_id=request.session_id,
            settings=settings,
        ),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    history = await chat_history.get_history(session_id)
    return {"session_id": session_id, "history": history}


@app.get("/models")
async def get_models(settings: Settings = Depends(get_settings)):
    assert llm is not None

    return {
        "id": settings.backend_type + " default model"
        if settings.model_path == ""
        else settings.model_path,
        "object": "model",
        "owned_by": "me",
        "permissions": [],
    }
