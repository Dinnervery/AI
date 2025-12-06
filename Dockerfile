FROM python:3.10-slim

# webrtcvad / pyaudio 등 빌드를 위한 패키지
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    portaudio19-dev \
    libasound2-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# FastAPI 서버 실행 (api_server.py 안의 app 사용)
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]