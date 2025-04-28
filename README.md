# AI Chat - Backend

2024 1학기 시작 졸업프로젝트 - **자동으로 맥락을 찾고 분류해주는 AI 기반 오픈 채팅 웹 어플리케이션**

### 사용 기술

#### Backend Server
- Node.js
- Express
- MongoDB
- Socket.IO
- Mongoose

#### AI Model Server
- Python
- Quart
- PyTorch
- Transformers
- OpenAI API
- Sentence Transformers
- scikit-learn

### 주요 기능

- 실시간 채팅
- 사용자 및 채팅방 관리
- 채팅 내용 자동 분류
- AI 기반 주제 요약
- 실시간 소켓 통신

### 실행 방법

#### 필수 요구사항
- Node.js 
- Python 3.8+
- MongoDB
- yarn 또는 npm

#### Backend Server 실행

1. 본 레포지토리를 clone 받습니다.
2. 루트 디렉토리에 `.env` 파일을 생성하고, 다음과 같은 내용으로 저장합니다:

```text
COOKIE_SECRET=<MongoDB 이름>                (가급적이면 aichat 이라는 이름으로 해주세요)
MONGO_ID=<Your MongoDB ID>
MONGO_PASSWORD=<Your MongoDB Password>
OPENAI_API_KEY=<Your-openai-api-key>
MONGODB_URI=mongodb:// <Your MongoDB ID> : <Your MongoDB Password> @localhost:27017/<MongoDB 이름>?authSource=admin&retryWrites=true&w=majority
```

3. 필요한 Node.js 패키지를 설치합니다:
```bash
yarn install
# 또는
npm install
```

4. MongoDB를 실행합니다:
```bash
mongod
```

5. 서버를 실행합니다:
```bash
yarn start
# 또는
npm start
```

#### AI Model Server 실행

1. Python 가상환경을 생성하고 활성화합니다:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. 필요한 Python 패키지를 설치합니다:
```bash
pip install -r requirements.txt
```

3. AI 모델 서버를 실행합니다:
```bash
python model.py
```

### API 엔드포인트

- `POST /user`: 새로운 사용자 등록
- `GET /chat`: 전체 채팅방 목록 조회
- `POST /chat`: 새로운 채팅방 생성
- `GET /chat/:id`: 특정 채팅방 입장
- `POST /chat/:id`: 채팅 메시지 전송
- `POST /chat/summary`: 채팅 내용 분류 및 요약

### 소켓 이벤트

- `connection`: 소켓 연결
- `join`: 채팅방 입장
- `message`: 메시지 전송
- `chat`: 새로운 채팅 메시지 수신
- `disconnect`: 소켓 연결 해제

### 데이터베이스 스키마

#### User
- nickname: String (required)
- userId: String (unique)

#### Room
- channelName: String (required)
- channelId: String

#### Chat
- room: ObjectId (ref: 'Room')
- nickname: String (required)
- content: String
- createdAt: Date
- topic: String
- embedding: [Number]

### 환경변수 설정

필요한 환경변수는 `.env` 파일에 설정해야 합니다:

- `COOKIE_SECRET`: 세션 암호화 키
- `MONGO_ID`: MongoDB 사용자 이름
- `MONGO_PASSWORD`: MongoDB 비밀번호
- `OPENAI_API_KEY`: OpenAI API 키
- `MONGODB_URI`: MongoDB 연결 문자열

### 개발 모드 실행

```bash
yarn dev
# 또는
npm run dev
```

개발 모드에서는 nodemon을 통해 파일 변경 시 자동으로 서버가 재시작됩니다.

# 채팅 분류 서비스

이 서비스는 채팅 메시지를 주제별로 분류하고 요약하는 기능을 제공합니다.

## 실제 서비스에 분류기 배포하기

현재 chat.travaa.site에 배포된 서비스에 이 분류기를 적용하려면 다음 단계를 따르세요:

### 1. 배포 옵션

#### 옵션 1: 독립적인 모델 API 서버로 배포

1. 서버에 이 저장소를 클론하고 필요한 모델 다운로드
```bash
git clone <repository-url> chat-classifier
cd chat-classifier
docker-compose --profile download up
```

2. 환경 변수 설정 (`.env` 파일 생성)
```
OPENAI_API_KEY=your_openai_api_key
```

3. 모델 API 서버 실행
```bash
docker-compose up -d
```

4. 기존 백엔드 서버의 환경 변수 설정
```
MODEL_SERVER_URL=http://<your-model-server-ip>:5000/predict
```

#### 옵션 2: 기존 서비스에 모델 코드 통합

1. 기존 메인 백엔드 서버에 모델 관련 파일 추가
   - `model.py`
   - 필요한 Python 패키지를 설치
   - 필요한 모델 파일 다운로드

2. NGINX 프록시 설정 수정
```nginx
# /model-api/ 경로의 요청을 Python 모델 서버로 전달
location /model-api/ {
    proxy_pass http://localhost:5000/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection 'upgrade';
    proxy_set_header Host $host;
    proxy_cache_bypass $http_upgrade;
}
```

### 2. 설정 및 테스트

1. 환경 변수 설정 확인
   - 메인 서버: `MODEL_SERVER_URL` 환경 변수가 올바르게 설정되었는지 확인
   - 모델 서버: `OPENAI_API_KEY` 환경 변수가 올바르게 설정되었는지 확인

2. 테스트
   - 채팅방에서 분류 기능 테스트
   - 분류 요청이 모델 서버로 올바르게 전달되는지 로그 확인

### 3. 모니터링 및 문제 해결

- 모델 서버 로그 확인: `docker-compose logs -f`
- 메인 서버 로그에서 분류 요청/응답 관련 오류 확인

## 문제 해결

- **모델 서버 연결 실패**: MODEL_SERVER_URL이 올바르게 설정되었는지 확인
- **메모리 부족**: 도커 컴포즈 파일에서 메모리 제한을 조정
- **느린 응답 시간**: 서버 사양 업그레이드 고려 또는 모델 최적화

## 최적화된 Docker 빌드 및 실행 방법

Docker 이미지 크기를 최적화하기 위해 아래 세 가지 방법을 사용할 수 있습니다.

### 1. 기본 방법 (볼륨을 이용한 모델 캐싱)

```bash
# Docker Compose를 사용하여 빌드 및 실행
docker-compose up -d
```

첫 실행 시 모델 파일이 다운로드되어 model_cache 볼륨에 저장됩니다. 이후에는 캐시된 모델을 사용합니다.

### 2. 사전 다운로드된 모델 사용 (권장)

모델을 미리 다운로드하여 이미지 빌드 시간과 크기를 줄입니다:

```bash
# 모델 다운로드 컨테이너 실행
docker run --rm -v $(pwd)/model_cache:/app/model_cache $(docker build -q -f Dockerfile.download .)

# 본 서비스 빌드 및 실행
docker-compose up -d
```

### 3. 경량화된 이미지 빌드 (가장 작은 이미지 크기)

경량화된 이미지를 빌드하려면 다음 명령어를 사용합니다:

```bash
# 이미지 빌드 (멀티스테이지 빌드)
docker build -t chat-classifier:optimized .

# 모델 캐시 볼륨과 함께 실행
docker run -d -p 5000:5000 -v $(pwd)/model_cache:/app/model_cache chat-classifier:optimized
```

## 모델 캐시

모델 파일은 다음 경로에 저장됩니다:
- `model_cache/`: 모델 파일이 저장되는 볼륨 디렉토리

## Docker 이미지 최적화 팁

1. 다단계 빌드 사용
2. 모델 파일을 볼륨으로 분리
3. `.dockerignore` 파일을 활용하여 불필요한 파일 제외
4. 경량화된 베이스 이미지 사용
5. 캐시 및 임시 파일 제거

## 모델 다운로드 및 도커 실행 방법

이 프로젝트는 도커 이미지를 통해 쉽게 실행할 수 있습니다. 단, 모델 파일이 큰 용량을 차지하므로 다음과 같은 방법으로 실행해야 합니다.

### 1. 모델 다운로드 (최초 1회)

최초 실행 시 모델 다운로드를 위해 다음 명령어를 실행합니다:

```bash
docker-compose run --profile download model-downloader
```

이 과정은 약 2-3GB의 모델 파일을 다운로드하므로 시간이 소요될 수 있습니다.

### 2. 서비스 실행

모델 다운로드가 완료된 후, 다음 명령어로 서비스를 실행합니다:

```bash
docker-compose up -d chat-classifier
```

### 주의사항

- 모델은 `model_cache` 볼륨에 저장되므로, 다운로드는 최초 1회만 필요합니다.
- 도커 이미지는 모델 파일을 포함하지 않아 용량이 작습니다 (약 500-600MB).
- `SKIP_MODEL_LOADING=true` 환경변수로 컨테이너 실행 시 미리 다운로드된 모델을 사용합니다.
- 개발 환경에서는 필요에 따라 `SKIP_MODEL_LOADING=false`로 설정하여 모델을 직접 다운로드할 수 있습니다.

### 도커 이미지 용량 최적화

도커 이미지 용량을 최소화하기 위한 방법:

1. `.dockerignore` 파일을 통해 불필요한 파일이 이미지에 포함되지 않도록 관리
2. 멀티 스테이지 빌드 사용으로 최종 이미지 크기 축소
3. 모델 파일은 별도 볼륨에 저장하여 이미지 크기와 분리

### 문제 해결

이미지 크기가 비정상적으로 큰 경우:

1. 도커 캐시 정리: `docker system prune -a`
2. 모든 중간 이미지 및 볼륨 확인: `docker images -a` 및 `docker volume ls`
3. 필요시 모델 볼륨 재생성: `docker volume rm model_cache` 후 모델 다운로더 실행
