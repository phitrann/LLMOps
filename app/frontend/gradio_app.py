# import gradio as gr
# from app.inference.pipeline import run_inference

# def predict(input_text):
#     return run_inference(input_text)

# iface = gr.Interface(
#     fn=predict,
#     inputs="text",
#     outputs="text",
#     title="LLM Inference",
#     description="Enter text to generate a response using LLM.",
# )

# if __name__ == "__main__":
#     iface.launch()


# import gradio as gr
# import httpx
# import asyncio
# import json

# API_URL = "http://localhost:8046"  # Adjust this if your API is hosted elsewhere

# async def chat_stream(message, history):
#     session_id = "gradio_session"  # You might want to generate a unique session ID

#     url = f"{API_URL}/chat/sse/"
#     headers = {'Accept': 'text/event-stream', 'Content-Type': 'application/json'}
#     data = {"session_id": session_id, "message": message}

#     # print(f"Sending request to {url} with data: {data}")

#     async with httpx.AsyncClient(timeout=None) as client:
#         try:
#             async with client.stream("POST", url, headers=headers, json=data) as response:
#                 response.raise_for_status()  # Raise an error for bad HTTP status
#                 full_response = ""
#                 async for chunk in response.aiter_text():
#                     if chunk:
#                         try:
#                             # print(f"Received chunk: {chunk.strip()}")
#                             data = json.loads(chunk.strip())
#                             if data['type'] == 'streaming':
#                                 full_response += data['value']
#                                 yield full_response
#                             elif data['type'] == 'end':
#                                 break
#                             elif data['type'] == 'error':
#                                 yield f"Error: {data['value']}"
#                                 break
#                         except json.JSONDecodeError:
#                             yield f"Error decoding response: {chunk.strip()}"
#         except httpx.RequestError as e:
#             yield f"Request error: {str(e)}"
#         except httpx.HTTPStatusError as e:
#             yield f"HTTP error: {str(e)}"


# def chat_stream_sync(message, history):
#     async_gen = chat_stream(message, history)
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     responses = []

#     try:
#         while True:
#             response = loop.run_until_complete(async_gen.__anext__())
#             responses.append(response)
#     except StopAsyncIteration:
#         pass

#     return responses

# iface = gr.ChatInterface(
#     chat_stream_sync,
#     chatbot=gr.Chatbot(height=400, ),
#     textbox=gr.Textbox(placeholder="Type your message here...", container=False, scale=7),
#     title="AI Chatbot",
#     description="Ask me anything!",
#     theme="soft",
#     examples=["Who's playing in the garden?", "Tell me about cats and dogs."],
#     cache_examples=False,
#     retry_btn=None,
#     undo_btn="Delete Last",
#     clear_btn="Clear",
# )

# if __name__ == "__main__":
#     iface.launch()





import gradio as gr
import httpx
import asyncio
import json

API_URL = "http://localhost:8046"  # Adjust this if your API is hosted elsewhere

# Step 1: Handle User Input
def user(user_message, history):
    # Append the user message to the history
    return "", history + [[user_message, None]]

# Step 2: Stream AI Response
async def bot(history):
    session_id = "gradio_session"  # You might want to generate a unique session ID
    
    url = f"{API_URL}/chat/sse/"
    headers = {'Accept': 'text/event-stream', 'Content-Type': 'application/json'}
    data = {"session_id": session_id, "message": history[-1][0]}  # Last user message

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("POST", url, headers=headers, json=data) as response:
                response.raise_for_status()  # Raise an error for bad HTTP status
                history[-1][1] = ""  # Start the AI's response as an empty string
                async for chunk in response.aiter_text():
                    if chunk:
                        try:
                            data = json.loads(chunk.strip())
                            if data['type'] == 'streaming':
                                # Update the bot's message in the history
                                history[-1][1] += data['value']
                                yield history
                            elif data['type'] == 'end':
                                break
                            elif data['type'] == 'error':
                                history[-1][1] = f"Error: {data['value']}"
                                yield history
                                break
                        except json.JSONDecodeError:
                            history[-1][1] = f"Error decoding response: {chunk.strip()}"
                            yield history
                            break
        except httpx.RequestError as e:
            history[-1][1] = f"Request error: {str(e)}"
            yield history
        except httpx.HTTPStatusError as e:
            history[-1][1] = f"HTTP error: {str(e)}"
            yield history

# Step 3: Download Chat History
def download_history(history):
    return gr.File.update(value=history, filename="chat_history.txt")

# Step 4: Clear Chat History
def clear_chat():
    return []

# Gradio Interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat History")
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your message here...",
                    show_label=False,
                    container=False,
                    scale=7,
                )
                submit_btn = gr.Button("Send")
            with gr.Row():
                clear = gr.Button("Clear Chat")
                download = gr.Button("Download Chat")
        with gr.Column(scale=1):
            gr.Markdown("## Chatbot Features")
            gr.Markdown("This chatbot can respond in real-time, providing streaming responses.")
            gr.Markdown("### Instructions:")
            gr.Markdown("1. Type a message and click 'Send' or press 'Enter'.")
            gr.Markdown("2. The chatbot will respond in real-time.")
            gr.Markdown("3. Use the buttons to clear or download the chat history.")

    # Handle user message submission
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    submit_btn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(clear_chat, None, chatbot, queue=False)
    download.click(download_history, chatbot)

# Launch the Gradio Interface
if __name__ == "__main__":
    demo.queue().launch()


