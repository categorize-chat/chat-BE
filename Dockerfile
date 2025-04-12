FROM python:3.9-slim AS base

# 기본 설정 및 패키지 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 난수 생성 관련 환경 변수 설정
ENV PYTHONHASHSEED=0
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache
ENV TORCH_HOME=/app/model_cache
ENV SKIP_MODEL_LOADING=false

# 기본 파이썬 패키지 설치 
RUN pip install --no-cache-dir --upgrade pip

FROM base AS builder

WORKDIR /app

# 필요한 파일 복사
COPY requirements.txt .

# 패키지 설치 (필요한 최소 패키지만 설치)
RUN pip install --no-cache-dir numpy==1.23.5 && \
    pip install --no-cache-dir torch==1.13.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir safetensors==0.4.3 && \
    pip install --no-cache-dir tokenizers==0.13.3 && \
    pip install --no-cache-dir scikit-learn==1.2.2 python-dotenv requests && \
    pip install --no-cache-dir transformers==4.30.2 && \
    pip install --no-cache-dir sentence-transformers==2.2.2 && \
    pip install --no-cache-dir quart && \
    find /usr/local/lib/python3.9/site-packages -name "*.pyc" -delete && \
    find /usr/local/lib/python3.9/site-packages -name "__pycache__" -exec rm -rf {} + && \
    rm -rf /root/.cache/pip

# 최종 이미지 - 실행 환경
FROM base AS final

WORKDIR /app

# 빌더 스테이지에서 설치된 패키지 복사
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 필요한 파일 복사
COPY model.py .
COPY .env .

# 포트 노출
EXPOSE 5000

# 모델 캐시 디렉토리 생성 및 권한 설정
RUN mkdir -p /app/model_cache && chmod 777 /app/model_cache

# 컨테이너 시작 시 실행할 명령
CMD ["python", "model.py"]