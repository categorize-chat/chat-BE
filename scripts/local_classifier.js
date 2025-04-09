const axios = require('axios');
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const dotenv = require('dotenv');

// .env 파일 로드
dotenv.config();

// 서버 URL 설정
const SERVER_URL = process.env.SERVER_URL || 'https://chat.travaa.site';
const MODEL_PORT = process.env.MODEL_PORT || 5000;
const API_PORT = process.env.API_PORT || 3000;

// 인증 토큰 설정 (사용자 로그인 후 얻은 토큰 사용)
let AUTH_TOKEN = process.env.AUTH_TOKEN;

// Express 앱 설정
const app = express();
app.use(cors());
app.use(bodyParser.json());

// 로그인 함수
async function login(email, password) {
  try {
    const response = await axios.post(`${SERVER_URL}/user/login`, {
      email,
      password
    });
    
    if (response.data.isSuccess) {
      AUTH_TOKEN = response.data.result.token;
      console.log('로그인 성공: 토큰이 설정되었습니다.');
      return AUTH_TOKEN;
    } else {
      console.error('로그인 실패:', response.data.message);
      return null;
    }
  } catch (error) {
    console.error('로그인 오류:', error.message);
    return null;
  }
}

// 채팅 데이터 가져오기 함수
async function fetchChats(roomId, howmany = 100) {
  try {
    if (!AUTH_TOKEN) {
      console.error('인증 토큰이 필요합니다.');
      return null;
    }

    const response = await axios.get(`${SERVER_URL}/chat/${roomId}?limit=${howmany}`, {
      headers: {
        Authorization: `Bearer ${AUTH_TOKEN}`
      }
    });

    if (response.data.isSuccess) {
      return response.data.result.messages;
    } else {
      console.error('채팅 데이터 가져오기 실패:', response.data.message);
      return null;
    }
  } catch (error) {
    console.error('채팅 데이터 가져오기 오류:', error.message);
    return null;
  }
}

// 모델 서버로 채팅 분류 요청 함수
async function classifyChats(roomId, chats) {
  try {
    const modelInput = {
      channelId: roomId,
      howmany: chats.length,
      chats: chats.map(chat => ({
        id: chat._id,
        nickname: chat.nickname || chat.user.nickname,
        content: chat.content,
        createdAt: chat.createdAt
      }))
    };

    console.log(`모델 서버에 분류 요청을 보냅니다: http://localhost:${MODEL_PORT}/predict`);
    const response = await axios.post(`http://localhost:${MODEL_PORT}/predict`, modelInput);
    console.log('모델 서버로부터 응답을 받았습니다.');
    return response.data;
  } catch (error) {
    console.error('채팅 분류 요청 오류:', error.message);
    return null;
  }
}

// 분류 결과를 서버로 전송하는 함수
async function sendClassificationResults(roomId, results) {
  try {
    if (!AUTH_TOKEN) {
      console.error('인증 토큰이 필요합니다.');
      return false;
    }

    const response = await axios.post(`${SERVER_URL}/chat/summary`, {
      channelId: roomId,
      topics: results.topics,
      summaries: results.summaries
    }, {
      headers: {
        Authorization: `Bearer ${AUTH_TOKEN}`
      }
    });

    if (response.data.isSuccess) {
      console.log('분류 결과가 성공적으로 서버로 전송되었습니다.');
      return true;
    } else {
      console.error('분류 결과 전송 실패:', response.data.message);
      return false;
    }
  } catch (error) {
    console.error('분류 결과 전송 오류:', error.message);
    return false;
  }
}

// 모델 서버 연결 확인 함수
async function checkModelServer() {
  try {
    console.log(`모델 서버 연결을 확인합니다: http://localhost:${MODEL_PORT}/health`);
    await axios.get(`http://localhost:${MODEL_PORT}/health`);
    console.log(`모델 서버(포트 ${MODEL_PORT})에 연결되었습니다.`);
    return true;
  } catch (error) {
    console.error(`모델 서버(포트 ${MODEL_PORT})에 연결할 수 없습니다.`);
    console.error('모델 서버가 실행 중인지 확인하세요.');
    console.error('Docker 컨테이너가 실행 중인지 확인하세요:');
    console.error('  1. docker ps 명령어로 실행 중인 컨테이너 확인');
    console.error('  2. docker-compose up -d 명령어로 Docker 컨테이너 시작');
    return false;
  }
}

// 주제 요약 요청 처리 API 엔드포인트
app.post('/chat/summary', async (req, res) => {
  try {
    console.log('주제 요약 요청 받음:', req.body);
    const { channelId, howmany } = req.body;
    
    if (!channelId) {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: "채널 ID가 필요합니다."
      });
    }
    
    // 채팅 데이터 가져오기
    const chats = await fetchChats(channelId, howmany || 100);
    if (!chats) {
      return res.status(500).json({
        isSuccess: false,
        code: 500,
        message: "채팅 데이터를 가져오는데 실패했습니다."
      });
    }
    
    // 모델 서버로 채팅 분류 요청
    const results = await classifyChats(channelId, chats);
    if (!results) {
      return res.status(500).json({
        isSuccess: false,
        code: 500,
        message: "채팅 분류에 실패했습니다."
      });
    }
    
    // 분류 결과를 서버로 전송
    const success = await sendClassificationResults(channelId, results);
    if (!success) {
      return res.status(500).json({
        isSuccess: false,
        code: 500,
        message: "분류 결과 전송에 실패했습니다."
      });
    }
    
    // 성공 응답
    return res.json({
      isSuccess: true,
      code: 200,
      message: "주제 요약에 성공했습니다.",
      result: results
    });
  } catch (error) {
    console.error('주제 요약 처리 오류:', error);
    return res.status(500).json({
      isSuccess: false,
      code: 500,
      message: "서버 오류가 발생했습니다.",
      error: error.message
    });
  }
});

// 서버 상태 확인 엔드포인트
app.get('/health', (req, res) => {
  return res.json({
    status: 'ok',
    message: '로컬 분류기 서버가 정상적으로 실행 중입니다.'
  });
});

// 메인 함수
async function main() {
  console.log('로컬 분류기 서비스를 시작합니다...');
  
  // 로그인 정보가 없다면 입력 받음
  if (!AUTH_TOKEN) {
    const email = process.env.EMAIL;
    const password = process.env.PASSWORD;
    
    if (!email || !password) {
      console.error('이메일과 비밀번호를 .env 파일에 설정해주세요.');
      process.exit(1);
    }
    
    AUTH_TOKEN = await login(email, password);
    if (!AUTH_TOKEN) {
      process.exit(1);
    }
  }

  // 모델 서버 연결 확인
  try {
    await checkModelServer();
  } catch (error) {
    console.warn('모델 서버 연결 확인 중 오류가 발생했습니다. 계속 진행합니다.');
  }

  // API 서버 시작
  return new Promise((resolve) => {
    const server = app.listen(API_PORT, () => {
      console.log(`로컬 API 서버가 http://localhost:${API_PORT}에서 실행 중입니다.`);
      console.log(`주제 요약 API 엔드포인트: http://localhost:${API_PORT}/chat/summary`);
      console.log(`프론트엔드에서 ${SERVER_URL} 대신 http://localhost:${API_PORT}으로 요청을 보내도록 설정하세요.`);
      console.log('프로그램이 실행 중입니다. 종료하려면 Ctrl+C를 누르세요.');
      
      // 서버 객체 반환
      resolve(server);
    });
  });
}

// 프로그램 시작 메시지
console.log('로컬 분류기 스크립트를 초기화합니다...');

// 프로그램 실행
main()
  .then(server => {
    // Node.js가 종료되지 않도록 이벤트 루프를 활성 상태로 유지
    console.log('서버가 성공적으로 시작되었습니다. 이벤트 루프를 유지합니다.');
    
    // SIGINT 처리 (Ctrl+C)
    process.on('SIGINT', () => {
      console.log('\n프로그램을 종료합니다.');
      server.close(() => {
        console.log('서버가 정상적으로 종료되었습니다.');
        process.exit(0);
      });
    });
    
    // 이벤트 루프를 강제로 활성 상태로 유지
    setInterval(() => {}, 1000 * 60 * 60); // 빈 함수를 1시간마다 실행
    
    // stdin을 열어 프로세스가 종료되지 않도록 함
    process.stdin.resume();
  })
  .catch(error => {
    console.error('프로그램 실행 오류:', error);
    process.exit(1);
  }); 