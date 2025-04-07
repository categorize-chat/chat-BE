# 로컬 채팅 분류 시스템 사용 가이드

이 가이드는 로컬 컴퓨터에서 채팅 분류 모델을 실행하는 방법을 설명합니다.

## 필요 사항

- Docker 및 Docker Compose가 설치되어 있어야 합니다.
- Node.js가 설치되어 있어야 합니다.
- chat.travaa.site 계정이 있어야 합니다.

## 설치 및 설정 방법

1. 이 저장소를 로컬 컴퓨터에 클론합니다.

2. `.env` 파일을 생성하고 다음 정보를 입력합니다:

```
# 서버 연결 정보
SERVER_URL=https://chat.travaa.site
MODEL_PORT=5000
API_PORT=3000

# 인증 정보 (다음 중 하나만 설정)
# 옵션 1: 로그인 정보
EMAIL=your_email@example.com
PASSWORD=your_password

# 옵션 2: 기존 토큰 (이미 로그인한 경우)
# AUTH_TOKEN=your_token_here

# 기타 설정
OPENAI_API_KEY=your_openai_api_key
```

3. 필요한 Node.js 패키지를 설치합니다:

```bash
npm install axios express cors body-parser dotenv
```

## 실행 방법

### 1. Docker를 사용하여 분류 모델 서버 시작

```bash
cd chat-BE
docker-compose up -d
```

### 2. 로컬 API 서버 시작

```bash
npm run local-classifier
```

## 프론트엔드 설정 변경

프론트엔드에서 API 요청을 로컬 API 서버로 리디렉션하기 위해 두 가지 방법이 있습니다:

### 방법 1: 브라우저 개발자 도구의 네트워크 요청 재지정 기능 사용

1. Chrome 브라우저의 개발자 도구를 엽니다 (F12 또는 우클릭 > 검사).
2. Network 탭에서 "Request blocking" 또는 "Request forwarding" 기능을 활성화합니다.
3. 다음 요청을 재지정합니다:
   - 원본: `https://chat.travaa.site/chat/summary`
   - 대상: `http://localhost:3000/chat/summary`

### 방법 2: 프론트엔드 환경 변수 수정 (개발 모드)

로컬에서 프론트엔드를 실행 중인 경우, `.env` 파일의 API 기본 URL을 변경할 수 있습니다:

```
VITE_API_BASE_URL=http://localhost:3000
```

## 작동 방식

1. 사용자가 UI에서 "주제 요약하기" 버튼을 클릭합니다.
2. 요청이 로컬 API 서버로 전송됩니다.
3. 로컬 API 서버는 chat.travaa.site에서 채팅 데이터를 가져옵니다.
4. 가져온 데이터는 로컬 도커 컨테이너의 분류 모델로 전송됩니다.
5. 분류 결과가 다시 chat.travaa.site로 전송되고, 프론트엔드에 표시됩니다.

## 문제 해결

### 모델 서버 로그 확인
```bash
docker-compose logs -f
```

### API 서버 오류
- 토큰 만료: 로그인 정보를 사용하여 새 토큰을 얻습니다.
- 연결 오류: 네트워크 연결과 서버 URL이 올바른지 확인합니다.
- 포트 충돌: API_PORT 환경 변수를 사용하여 다른 포트를 지정합니다.

### CORS 오류
로컬 API 서버에서 CORS 설정이 활성화되어 있지만, 문제가 발생하면 브라우저 확장 프로그램을 사용하여 CORS를 우회할 수 있습니다.

## 보안 참고 사항

- `.env` 파일에 중요한 인증 정보가 포함되어 있으므로 안전하게 보관하세요.
- 개인 컴퓨터에서만 이 시스템을 실행하는 것을 권장합니다.
- 공유 네트워크나 공용 환경에서 실행할 경우 보안에 주의하세요. 