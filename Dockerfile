FROM python:3.9-slim AS builder

WORKDIR /app

# 빌드에 필요한 패키지만 설치
RUN pip install --no-cache-dir --upgrade pip

# 필요한 파일 복사
COPY requirements.txt .

# 패키지 설치 및 캐시 정리 (한 RUN 명령으로 통합하여 레이어 수 감소)
RUN pip install --no-cache-dir torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir quart tokenizers \
    && pip install --no-cache-dir tensorflow \
    && pip install --no-cache-dir numpy==1.23.5 \
    && pip install --no-cache-dir scikit-learn python-dotenv requests \
    && pip install --no-cache-dir safetensors>=0.4.3 \
    && pip install --no-cache-dir transformers==4.36.2 \
    && pip install --no-cache-dir sentence-transformers==2.2.2 \
    && find /usr/local/lib/python3.9/site-packages -name "*.pyc" -delete \
    && find /usr/local/lib/python3.9/site-packages -name "__pycache__" -exec rm -rf {} + \
    && rm -rf /root/.cache/pip

# 경량화된 최종 이미지
FROM python:3.9-slim

WORKDIR /app

# 난수 생성 관련 환경 변수 설정
ENV PYTHONHASHSEED=0
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# 빌더 스테이지에서 설치된 패키지 복사
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 필요한 파일 복사
COPY model.py .
COPY .env .

# 포트 노출
EXPOSE 5000

# 불필요한 캐시 및 임시 파일 제거
RUN apt-get update \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache \
    && rm -rf /tmp/*

# 모델 캐시 디렉토리 생성 및 권한 설정
RUN mkdir -p /app/model_cache \
    && chmod 777 /app/model_cache

# 컨테이너 시작 시 실행할 명령
CMD ["python", "model.py"]