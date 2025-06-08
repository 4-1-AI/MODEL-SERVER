# 1. 베이스 이미지: 가벼운 Python + CUDA 필요시 따로 조정 가능
FROM python:3.10-slim

# 2. 작업 디렉토리
WORKDIR /app

# 3. 시스템 패키지 설치 (OpenCV 등 빌드에 필요)
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 4. 필수 파일 복사
COPY requirements.txt .

RUN git clone https://github.com/ultralytics/yolov5.git /app/yolov5
ENV PYTHONPATH="/app/yolov5"

# 5. 라이브러리 설치
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# 6. 전체 프로젝트 복사
COPY . .

# 7. FastAPI 앱 실행 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
