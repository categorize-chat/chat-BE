#!/bin/bash

# 현재 디렉토리로 이동
cd "$(dirname "$0")/.."
echo "현재 디렉토리: $(pwd)"

# 테스트에 필요한 패키지 설치
echo "테스트 패키지 설치 중..."
npm install --save-dev socket.io-client uuid autocannon

echo "설치 완료!" 