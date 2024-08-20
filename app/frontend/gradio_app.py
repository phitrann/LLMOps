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
    
    timeout = httpx.Timeout(10.0, connect=5.0, read=None)  # Set `read` to None for no timeout during streaming
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with client.stream("POST", url, headers=headers, json=data) as response:
                response.raise_for_status()  # Raise an error for bad HTTP status
                history[-1][1] = ""  # Start the AI's response as an empty string
                async for chunk in response.aiter_text():
                    chunk = chunk.strip()
                    if not chunk:
                        continue

                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        history[-1][1] = f"Error decoding response: {chunk}"
                        yield history
                        break
                    
                    # Stream the response as it comes in
                    if data['type'] == 'streaming':
                        history[-1][1] += data['value']
                        yield history
                    elif data['type'] == 'end':
                        break
                    elif data['type'] == 'error':
                        history[-1][1] = f"Error: {data['value']}"
                        yield history
                        break
                    # asyncio.sleep(0.1)
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

# Helper function to run async bot
async def run_bot(history):
    async for updated_history in bot(history):
        return updated_history

# # Gradio Interface
# with gr.Blocks() as demo:
#     with gr.Row():
#         with gr.Column(scale=3):
#             chatbot = gr.Chatbot(label="Chat History")
#             with gr.Row():
#                 msg = gr.Textbox(
#                     placeholder="Type your message here...",
#                     show_label=False,
#                     container=False,
#                     scale=7,
#                 )
#                 submit_btn = gr.Button("Send")
#             with gr.Row():
#                 clear = gr.Button("Clear Chat")
#                 download = gr.Button("Download Chat")
#         with gr.Column(scale=1):
#             gr.Markdown("## Chatbot Features")
#             gr.Markdown("This chatbot can respond in real-time, providing streaming responses.")
#             gr.Markdown("### Instructions:")
#             gr.Markdown("1. Type a message and click 'Send' or press 'Enter'.")
#             gr.Markdown("2. The chatbot will respond in real-time.")
#             gr.Markdown("3. Use the buttons to clear or download the chat history.")

#      # Handle user message submission
#     msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
#         # lambda history: asyncio.run(run_bot(history)), chatbot, chatbot
#         bot, chatbot, chatbot
#     )
#     submit_btn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
#         # lambda history: asyncio.run(run_bot(history)), chatbot, chatbot
#         bot, chatbot, chatbot
#     )
#     clear.click(clear_chat, None, chatbot, queue=False)
#     download.click(download_history, chatbot)

# Gradio Interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("# 💬 Interactive Chatbot Interface")
            chatbot = gr.Chatbot(label="Chat History", height=600)
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your message here...",
                    show_label=False,
                    container=False,
                    scale=8,
                )
                submit_btn = gr.Button("🚀 Send", variant="primary")
                
            with gr.Row():
                clear = gr.Button("🗑️ Clear Chat", variant="secondary")
                download = gr.Button("💾 Download Chat", variant="secondary")
            
            with gr.Accordion("Additional Tools", open=False):
                file_upload = gr.File(label="Upload a file", file_count="single")
                search_box = gr.Textbox(label="Search in Chat History", placeholder="Type keywords here...")
                search_btn = gr.Button("🔍 Search", variant="primary")
            
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("📄 Instructions"):
                    gr.Markdown("""
                    ### How to use this Chatbot:
                    1. **Send a message**: Type your message in the input box and click 'Send'.
                    2. **Real-time response**: The chatbot will respond in real-time.
                    3. **File Upload**: Use the file upload tool to send files (coming soon!).
                    4. **Search**: You can search through the chat history using the search tool.
                    """)
                
                with gr.TabItem("💡 Tips & Tricks"):
                    gr.Markdown("""
                    - Use the 'Clear Chat' button to reset the conversation.
                    - Download your chat history for later review.
                    """)
                    
                with gr.TabItem("❓ Help"):
                    gr.Markdown("""
                    If you need help, please refer to the official documentation or contact support.
                    """)
                    
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


