# from langchain_community.chat_models.ollama import ChatOllama
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# import os

# os.environ["NO_PROXY"] = "172.16.87.75"

# def get_chain(model, prompt):
#     # This is the language model we'll use.
#     # We'll talk more about what we're doing in the next section.
#     llm = ChatOllama(model=model, base_url="http://172.16.87.75:11434")
#     prompt_template = ChatPromptTemplate.from_template((prompt))
#     chain = prompt_template | llm | StrOutputParser()

#     return chain

# # Example
# question = "What is the capital of France?"
# context = "France is a country in Europe. It has several large cities including Paris, Lyon, and Marseille. Paris is known for the Eiffel Tower."

# chain = get_chain(
#     model="phi3",
#     prompt="Please answer the following question based on the provided `context` that follows the question.\n"
#     "Think step by step before coming to answer. If you do not know the answer then just say 'I do not know'\n"
#     "question: {question}\n"
#     "context: ```{context}```\n",
# )

# # Use the chain to generate the answer
# result = chain.invoke({"question": question, "context": context})

# # Print the result
# print(result)


from pprint import pprint
import httpx


async def stream_chat_sse_response():
    async with httpx.AsyncClient() as client:
        # Replace this URL with your FastAPI server URL
        url = "http://localhost:8046/chat/sse/"
        
        # Data to send in the request
        data = {
            "session_id": "test_session",
            "message": "Who is the president of the United States?"
        }

        # Send the POST request and stream the response
        async with client.stream("POST", url, json=data) as response:
            async for chunk in response.aiter_text():
                print(chunk, end="\n")

# Run the functions to test streaming
import asyncio
# Stream SSE response
print("\n\nStreaming SSE response:")
asyncio.run(stream_chat_sse_response())