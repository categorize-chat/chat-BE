FROM python:3.9-slim AS builder

WORKDIR /app

# 빌드에 필요한 패키지만 설치
RUN pip install --no-cache-dir --upgrade pip

# 필요한 파일 복사
COPY requirements.txt .

# 패키지 설치 및 캐시 정리
RUN pip install --no-cache-dir torch==2.0.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir quart && \
    pip install --no-cache-dir tokenizers && \
    pip install --no-cache-dir tensorflow && \
    pip install --no-cache-dir scikit-learn python-dotenv openai numpy asyncio tqdm regex requests pyyaml && \
    pip install --no-cache-dir --no-deps transformers && \
    pip install --no-cache-dir --no-deps sentence-transformers && \
    find /usr/local/lib/python3.9/site-packages -name "*.pyc" -delete && \
    find /usr/local/lib/python3.9/site-packages -name "__pycache__" -exec rm -rf {} +

# 경량화된 최종 이미지
FROM python:3.9-slim

WORKDIR /app

# 빌더 스테이지에서 설치된 패키지 복사
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 필요한 파일 복사
COPY model.py .
COPY .env .

# 포트 노출
EXPOSE 5000

# 불필요한 캐시 및 임시 파일 제거
RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache

# 컨테이너 시작 시 실행할 명령
CMD ["python", "model.py"]