#!/bin/bash

echo "모델 서버 빌드 및 실행 시작..."
docker-compose up -d --build
echo "모델 서버가 백그라운드에서 실행 중입니다."
echo "로그 확인: docker-compose logs -f"
echo "서버 중지: docker-compose down" 