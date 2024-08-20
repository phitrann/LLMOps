# LLM Inference Project

This project uses FastAPI for the backend API, Gradio for the frontend, and Hugging Face's Transformers library for LLM inference.

## Introduction


## Project Structure
```
/LLMOps
├── /app
│   ├── /api
│   │   ├── __init__.py
│   │   ├── main.py               # Điểm vào của FastAPI, định nghĩa các routes chính
│   │   ├── models.py             # Định nghĩa các schema dữ liệu cho API
│   │   ├── config.py             # Cấu hình cho ứng dụng (env variables, constants, etc.)
│   │   └── utils.py              # Các hàm tiện ích dùng chung
│   ├── /inference
│   │   ├── __init__.py
│   │   ├── model.py              # Chứa logic để tải và chạy inference mô hình LLM
│   │   ├── tokenizer.py          # Xử lý mã hóa và giải mã văn bản
│   │   └── pipeline.py           # Định nghĩa pipeline xử lý từ input đến output
│   ├── /frontend
│   │   ├── __init__.py
│   │   ├── gradio_app.py         # Cấu hình và chạy Gradio app
│   │   └── components.py         # Các thành phần UI của Gradio
│   ├── /db
│   │   ├── __init__.py
│   │   ├── database.py           # Kết nối và tương tác với database (PostgreSQL/MongoDB)
│   │   ├── models.py             # Định nghĩa các models cho ORM
│   │   └── history.py            # Xử lý lưu trữ và truy xuất lịch sử truy vấn
│   ├── /tests
│   │   ├── test_api.py           # Unit test cho API
│   │   ├── test_inference.py     # Unit test cho mô hình và pipeline
│   │   ├── test_gradio.py        # Unit test cho Gradio frontend
│   │   └── test_db.py            # Unit test cho database và lịch sử
│   └── main.py                   # Điểm vào chính cho việc chạy toàn bộ ứng dụng
├── /scripts
│   ├── download_model.py         # Script tải về mô hình từ Hugging Face
│   ├── start_server.sh           # Script để khởi động server (FastAPI, Gradio)
│   └── init_db.py                # Script khởi tạo và thiết lập database
├── /docker
│   ├── Dockerfile                # Định nghĩa Dockerfile cho ứng dụng
│   ├── docker-compose.yml        # Định nghĩa Docker Compose cho toàn bộ hệ thống
│   └── /k8s
│       ├── deployment.yaml       # Cấu hình Kubernetes Deployment
│       └── service.yaml          # Cấu hình Kubernetes Service
├── .env                          # Tập tin cấu hình biến môi trường
├── .gitignore                    # Định nghĩa các file và thư mục cần bỏ qua khi commit git
├── requirements.txt              # Liệt kê các dependencies Python cần thiết
└── README.md                     # Hướng dẫn cài đặt và sử dụng dự án
```

__Main components__:
1. __/app__: Thư mục chính chứa toàn bộ mã nguồn của ứng dụng.
    - __/api__: Xử lý các endpoint của FastAPI, cung cấp API cho ứng dụng.
    - __/inference__: Chứa logic liên quan đến mô hình LLM, từ việc tải mô hình, chạy inference đến pipeline xử lý.
    - __/frontend__: Chứa cấu hình và logic liên quan đến Gradio, nơi mà bạn sẽ thiết kế giao diện người dùng.
    - __/db__: Quản lý kết nối với database, định nghĩa các model và thao tác với lịch sử truy vấn.
    - __/tests__: Chứa các unit test cho từng thành phần của ứng dụng.
2. __/scripts__: Các script tiện ích để quản lý dự án như tải mô hình, khởi động server, và khởi tạo database.

3. __/docker__: Chứa các file cấu hình liên quan đến Docker và Kubernetes để đóng gói và triển khai ứng dụng.

4. __.env__: File cấu hình chứa các biến môi trường, như thông tin kết nối database, API keys,...

5. __requirements.txt__: Danh sách các thư viện Python cần thiết cho dự án.

6. __README.md__: Hướng dẫn sử dụng dự án, bao gồm cách cài đặt, chạy ứng dụng, và cách triển khai.

## Models
- The model used in this project is `microsoft/Phi-3-mini-4k-instruct`. Phi-3 Mini is a 3.8B parameters, lightweight, state-of-the-art open model trained with the Phi-3 datasets that includes both synthetic data and the filtered publicly available websites data with a focus on high-quality and reasoning dense properties.


## Tech Stacks
- FastAPI: A modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.
- Gradio: A Python library that allows you to quickly create UIs for your machine learning models.
- Langchain: A Python library that provides components for building conversational AI systems.
- Ollama: Serving for LLM inference.


## Requirements
### Prequisites
1. Clone the repository
```bash
git clone https://github.com/phitrann/LLMOps.git
```

2. Set up Ollama server
```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

3. Pull the Phi 3 Mini model
```bash
docker exec -it ollama ollama run phi3:instruct
```

4. Set up the MongoDB database
```bash
docker compose -f docker/docker-compose.mongo.yaml up -d
```


### Installation

#### Manual


1. Install dependencies
```bash
pip install -e .
pip install -r requirements.txt
```

2. Download the model (Optional)
```bash
python scripts/download_model.py
```

3. Start the MongoDB service
```
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

4. Start the server
```bash
# Start FastAPI server
uvicorn app.api.main:app --host 0.0.0.0 --port 8046 --reload

# Start Gradio app
python -m app.frontend.gradio_app # gradio app/frontend/gradio_app.py
```

#### Docker Compose (Not working yet)

```bash
docker compose up -d
```

### Test the APIs

1. Send a message to the chatbot
```
curl --no-buffer -X 'POST' 'http://localhost:8046/chat' -H 'accept: text/plain' -H 'Content-Type: application/json' -d '{"session_id": "session_1","message": "who'\''s playing in the river?"}'
```

2. Send a message to the chatbot through server sent events
```
curl --no-buffer -X 'POST' 'http://localhost:8046/chat/sse/' -H 'accept: text/event-stream' -H 'Content-Type: application/json' -d '{"session_id": "session_2", "message": "who'\''s playing in the garden?"}'
```


## Usage

- The API will be available at `http://localhost:8046`.
- The Gradio interface will be available at `http://localhost:7860`.





