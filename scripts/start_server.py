def main():
    # Start FastAPI server
    import os
    os.system("uvicorn app.api.main:app --host 0.0.0.0 --port 8019 &")
    
    # Start Gradio app
    from app.frontend.gradio_app import iface
    iface.launch()
