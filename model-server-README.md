# 로컬 모델 서버 설정 가이드

이 가이드는 채팅 토픽 분류 모델을 로컬 컴퓨터에서 Docker를 통해 실행하는 방법을 설명합니다.

## 사전 요구사항

- [Docker](https://www.docker.com/products/docker-desktop/) 설치
- [Docker Compose](https://docs.docker.com/compose/install/) 설치 (최신 Docker Desktop에는 포함되어 있음)

## 설정 방법

1. `.env` 파일이 프로젝트 루트에 있는지 확인하세요. 이 파일에는 필요한 API 키와 환경 변수가 포함되어 있어야 합니다.

   ```
   OPENAI_API_KEY=your_api_key_here
   ```

2. 아래 명령을 실행하여 서버를 시작하세요:

   ```bash
   ./run-model-server.sh
   ```

   또는 직접 Docker Compose 명령을 실행할 수 있습니다:

   ```bash
   docker-compose up -d --build
   ```

3. 서버가 시작되면 다음 주소로 접근할 수 있습니다:
   - http://localhost:5000

## 로그 확인

```bash
docker-compose logs -f
```

## 서버 중지

```bash
docker-compose down
```

## 서버 재시작

```bash
docker-compose restart
```

## 백엔드 서버 연결 설정

기존 백엔드에서 AWS 대신 로컬 모델 서버를 사용하려면 환경 변수 설정을 변경하세요:

```
MODEL_API_URL=http://localhost:5000
```

## 문제 해결

- 모델 로딩에 시간이 걸릴 수 있습니다. 첫 시작 시 몇 분 정도 기다려주세요.
- GPU가 있는 경우 더 빠른 처리를 위해 Docker에서 GPU를 사용하도록 설정할 수 있습니다. 