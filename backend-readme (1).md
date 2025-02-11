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
# 외부 패키지 설치
pip install quart tensorflow torch transformers sentence-transformers scikit-learn python-dotenv openai numpy
```

> 📝 프로젝트에서는 Python 기본 내장 라이브러리인 re(정규표현식), os(운영체제), asyncio(비동기 처리), json, traceback, datetime, copy 등도 사용합니다. 이들은 Python 설치 시 기본으로 제공되므로 별도 설치가 필요하지 않습니다.


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
