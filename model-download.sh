#!/bin/bash

# 모델 다운로더 서비스 구성 파일 생성
cat > docker-compose.download.yml << EOF
version: '3'

services:
  model-downloader:
    build:
      context: .
      dockerfile: Dockerfile.download
    image: chat-classifier-downloader:latest
    volumes:
      - model_cache:/app/model_cache
    environment:
      - PYTHONUNBUFFERED=1

volumes:
  model_cache:
    driver: local
EOF

echo "모델 다운로더 실행 중..."
docker-compose -f docker-compose.download.yml up --build
echo "모델 다운로드 완료. model_cache 볼륨에 저장되었습니다."
echo "이제 다음 명령으로 주 서비스를 실행하세요: docker-compose up -d" 