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

    const response = await axios.post(`http://localhost:${MODEL_PORT}/predict`, modelInput);
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

// 메인 함수
async function main() {
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

  // API 서버 시작
  app.listen(API_PORT, () => {
    console.log(`로컬 API 서버가 http://localhost:${API_PORT}에서 실행 중입니다.`);
    console.log(`주제 요약 API 엔드포인트: http://localhost:${API_PORT}/chat/summary`);
    console.log(`프론트엔드에서 ${SERVER_URL} 대신 http://localhost:${API_PORT}으로 요청을 보내도록 설정하세요.`);
  });
}

// 프로그램 실행
main().catch(error => {
  console.error('프로그램 실행 오류:', error);
}); 