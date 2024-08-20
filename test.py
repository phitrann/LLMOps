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

#########################################################################################

# from pprint import pprint
# import httpx
# import asyncio

# async def stream_chat_sse_response():
#     timeout = httpx.Timeout(10.0, connect=5.0, read=None)  # Set `read` to None for no timeout during streaming
#     try:
#         async with httpx.AsyncClient(timeout=timeout) as client:
#             # Replace this URL with your FastAPI server URL
#             url = "http://localhost:8046/chat/sse/"
            
#             # Data to send in the request
#             data = {
#                 "session_id": "test_session",
#                 "message": "How to cook Bun Cha in Vietnam?"
#             }

#             # Send the POST request and stream the response
#             async with client.stream("POST", url, json=data) as response:
#                 # Check if the response status is 200 OK
#                 response.raise_for_status()
#                 async for chunk in response.aiter_text():
#                     if chunk:
#                         print(chunk, end="\n")
#     except httpx.HTTPStatusError as exc:
#         print(f"HTTP error occurred: {exc}")
#     except httpx.RequestError as exc:
#         print(f"An error occurred while requesting: {exc}")
#     except httpx.ReadTimeout:
#         print("The request timed out while reading the response.")
#     except Exception as exc:
#         print(f"An unexpected error occurred: {exc}")

# # Run the function to test streaming
# print("\n\nStreaming SSE response:")
# asyncio.run(stream_chat_sse_response())
