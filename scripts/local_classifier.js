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
    console.log(`로그인 시도 중: ${SERVER_URL}/user/login`);
    console.log(`이메일: ${email}`);
    console.log(`비밀번호: ${'*'.repeat(password.length)}`);
    
    const response = await axios.post(`${SERVER_URL}/user/login`, {
      email,
      password
    });
    
    console.log('서버 응답 상태:', response.status);
    console.log('서버 응답 데이터 구조:', Object.keys(response.data));
    console.log('서버 응답 데이터:', JSON.stringify(response.data, null, 2));
    
    // 여러 가능한 응답 구조 처리
    if (response.status === 200) {
      // 케이스 1: isSuccess 필드 확인
      if (response.data.isSuccess === true && response.data.result && response.data.result.token) {
        console.log('케이스 1: isSuccess 필드 확인 - 성공');
        return response.data.result.token;
      }
      
      // 케이스 1-1: accessToken 필드 확인 (추가)
      if (response.data.isSuccess === true && response.data.result && response.data.result.accessToken) {
        console.log('케이스 1-1: accessToken 필드 확인 - 성공');
        return response.data.result.accessToken;
      }
      
      // 케이스 2: success 필드 확인
      if (response.data.success === true && response.data.token) {
        console.log('케이스 2: success 필드 확인 - 성공');
        return response.data.token;
      }
      
      // 케이스 3: token 필드 직접 확인
      if (response.data.token) {
        console.log('케이스 3: token 필드 직접 확인 - 성공');
        return response.data.token;
      }
      
      // 케이스 3-1: accessToken 필드 직접 확인 (추가)
      if (response.data.accessToken) {
        console.log('케이스 3-1: accessToken 필드 직접 확인 - 성공');
        return response.data.accessToken;
      }
      
      // 케이스 4: data 내부에 token 필드 확인
      if (response.data.data && response.data.data.token) {
        console.log('케이스 4: data 내부에 token 필드 확인 - 성공');
        return response.data.data.token;
      }
      
      // 케이스 4-1: data 내부에 accessToken 필드 확인 (추가)
      if (response.data.data && response.data.data.accessToken) {
        console.log('케이스 4-1: data 내부에 accessToken 필드 확인 - 성공');
        return response.data.data.accessToken;
      }
      
      // 케이스 5: JWT 형식 문자열인 경우
      if (typeof response.data === 'string' && response.data.startsWith('ey')) {
        console.log('케이스 5: JWT 형식 문자열 확인 - 성공');
        return response.data;
      }
      
      // 응답 구조가 예상과 다름
      console.error('로그인은 성공했지만 토큰을 찾을 수 없습니다.');
      console.error('응답 구조:', JSON.stringify(response.data, null, 2));
      console.error('서버 응답에 맞게 코드를 수정해주세요.');
      return null;
    } else {
      console.error('로그인 실패:', response.statusText);
      return null;
    }
  } catch (error) {
    console.error('로그인 오류:', error.message);
    if (error.response) {
      console.error('서버 응답 데이터:', JSON.stringify(error.response.data, null, 2));
      console.error('서버 응답 상태:', error.response.status);
      console.error('서버 응답 헤더:', JSON.stringify(error.response.headers, null, 2));
    } else if (error.request) {
      console.error('요청은 보냈지만 응답을 받지 못했습니다. 서버가 실행 중인지 확인하세요.');
      console.error('서버 URL:', SERVER_URL);
    } else {
      console.error('요청 설정 중 오류가 발생했습니다:', error.message);
    }
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
    
    // 여러 가능한 모델 서버 URL 시도
    let error;
    
    // 방법 1: 로컬호스트 시도
    try {
      const response = await axios.post(`http://localhost:${MODEL_PORT}/predict`, modelInput);
      console.log('모델 서버로부터 응답을 받았습니다.');
      return response.data;
    } catch (err) {
      console.log(`로컬호스트 연결 실패: ${err.message}`);
      error = err;
    }
    
    // 방법 2: 컨테이너 이름 시도
    try {
      const response = await axios.post(`http://chat-classifier:${MODEL_PORT}/predict`, modelInput);
      console.log('컨테이너 이름으로 모델 서버로부터 응답을 받았습니다.');
      return response.data;
    } catch (err) {
      console.log(`컨테이너 이름 연결 실패: ${err.message}`);
    }
    
    // 방법 3: Docker 호스트 IP 시도
    try {
      const response = await axios.post(`http://host.docker.internal:${MODEL_PORT}/predict`, modelInput);
      console.log('Docker 호스트 IP로 모델 서버로부터 응답을 받았습니다.');
      return response.data;
    } catch (err) {
      console.log(`Docker 호스트 IP 연결 실패: ${err.message}`);
    }
    
    // 모든 방법 실패
    throw error || new Error('모든 연결 방법이 실패했습니다.');
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
    
    // 여러 가능한 모델 서버 URL 시도
    let error;
    
    // 방법 1: 로컬호스트 시도
    try {
      await axios.get(`http://localhost:${MODEL_PORT}/health`);
      console.log(`모델 서버(포트 ${MODEL_PORT})에 연결되었습니다.`);
      return true;
    } catch (err) {
      console.log(`로컬호스트 연결 실패: ${err.message}`);
      error = err;
    }
    
    // 방법 2: 컨테이너 이름 시도
    try {
      await axios.get(`http://chat-classifier:${MODEL_PORT}/health`);
      console.log(`컨테이너 이름으로 모델 서버(포트 ${MODEL_PORT})에 연결되었습니다.`);
      return true;
    } catch (err) {
      console.log(`컨테이너 이름 연결 실패: ${err.message}`);
    }
    
    // 방법 3: Docker 호스트 IP 시도
    try {
      await axios.get(`http://host.docker.internal:${MODEL_PORT}/health`);
      console.log(`Docker 호스트 IP로 모델 서버(포트 ${MODEL_PORT})에 연결되었습니다.`);
      return true;
    } catch (err) {
      console.log(`Docker 호스트 IP 연결 실패: ${err.message}`);
    }
    
    // 모든 방법 실패
    throw error || new Error('모든 연결 방법이 실패했습니다.');
  } catch (error) {
    console.error(`모델 서버(포트 ${MODEL_PORT})에 연결할 수 없습니다. 오류: ${error.message}`);
    console.error('모델 서버가 실행 중인지 확인하세요.');
    console.error('Docker 컨테이너가 실행 중인지 확인하세요:');
    console.error('  1. docker ps 명령어로 실행 중인 컨테이너 확인');
    console.error('  2. docker-compose up -d 명령어로 Docker 컨테이너 시작');
    
    // 모델 서버가 없어도 계속 진행하지만 사용자에게 알림
    console.log('모델 서버가 없어도 로컬 API 서버는 시작됩니다. 하지만 분류 기능은 동작하지 않을 수 있습니다.');
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

// 포트 사용 가능 여부 확인 및 서버 시작 함수
async function startServer(port) {
  return new Promise((resolve, reject) => {
    try {
      const svr = app.listen(port, () => {
        console.log(`로컬 API 서버가 http://localhost:${port}에서 실행 중입니다.`);
        console.log(`주제 요약 API 엔드포인트: http://localhost:${port}/chat/summary`);
        console.log(`프론트엔드에서 ${SERVER_URL} 대신 http://localhost:${port}으로 요청을 보내도록 설정하세요.`);
        resolve(svr);
      });
      
      svr.on('error', (err) => {
        if (err.code === 'EADDRINUSE') {
          console.warn(`포트 ${port}가 이미 사용 중입니다.`);
          resolve(null); // 다른 포트 시도를 위해 null 반환
        } else {
          console.error('서버 시작 오류:', err.message);
          reject(new Error(`서버 시작 오류: ${err.message}`));
        }
      });
    } catch (error) {
      reject(error);
    }
  });
}

// 메인 함수
async function main() {
  try {
    console.log('로컬 분류기 서비스를 시작합니다...');
    
    // 인증 토큰 처리
    if (!AUTH_TOKEN) {
      console.log('AUTH_TOKEN이 없습니다. .env 파일의 인증 정보를 사용하여 로그인합니다.');
      const email = process.env.EMAIL;
      const password = process.env.PASSWORD;
      
      if (!email || !password) {
        console.error('이메일과 비밀번호를 .env 파일에 설정해주세요.');
        console.error('다음을 .env 파일에 추가해주세요:');
        console.error('EMAIL=your-email@example.com');
        console.error('PASSWORD=your-password');
        return null;
      }
      
      console.log(`서버(${SERVER_URL})에 로그인을 시도합니다...`);
      const token = await login(email, password);
      
      if (!token) {
        console.error('로그인에 실패했습니다. 서비스를 시작할 수 없습니다.');
        console.error('이메일과 비밀번호를 확인해주세요.');
        return null;
      }
      
      // 토큰 설정
      AUTH_TOKEN = token;
      console.log('로그인 성공: 토큰이 설정되었습니다.');
    } else {
      console.log('기존 토큰을 사용합니다.');
    }

    // 모델 서버 연결 확인
    let modelServerAvailable = false;
    try {
      modelServerAvailable = await checkModelServer();
    } catch (error) {
      console.error('모델 서버 연결 확인 중 오류가 발생했습니다:', error.message);
      console.warn('모델 서버가 없어도 로컬 API 서버는 시작됩니다.');
    }

    // API 서버 시작
    console.log('로컬 API 서버를 시작합니다...');
    let server = null;
    let portToUse = API_PORT;
    
    // 여러 포트 시도
    try {
      server = await startServer(portToUse);
      
      // 원래 포트를 사용할 수 없는 경우 다른 포트 시도
      if (!server) {
        const alternativePorts = [3001, 3002, 3003, 8080, 8081];
        
        for (const port of alternativePorts) {
          console.log(`대체 포트 ${port}로 서버 시작을 시도합니다...`);
          server = await startServer(port);
          
          if (server) {
            portToUse = port;
            break;
          }
        }
      }
      
      if (!server) {
        console.error('사용 가능한 포트를 찾을 수 없습니다.');
        console.error('다음 포트를 시도했습니다: 3000, 3001, 3002, 3003, 8080, 8081');
        console.error('.env 파일에서 API_PORT 값을 다른 값으로 변경하거나 기존 프로세스를 종료해주세요.');
        return null;
      }
      
      console.log(`API 서버가 포트 ${portToUse}에서 성공적으로 시작되었습니다.`);
      return server;
    } catch (error) {
      console.error('API 서버 시작 중 오류:', error.message);
      return null;
    }
  } catch (error) {
    console.error('메인 함수 실행 중 오류:', error.message);
    console.error('스택 트레이스:', error.stack);
    return null;
  }
}

// 프로그램 시작 메시지
console.log('로컬 분류기 스크립트를 초기화합니다...');

// Windows PowerShell에서 실행을 위한 메인 함수 호출
(async () => {
  try {
    const server = await main();
    
    if (!server) {
      console.error('서버 시작에 실패했습니다.');
      console.error('위의 오류 메시지를 확인하여 문제를 해결하세요.');
      console.error('프로그램이 10초 후에 종료됩니다...');
      
      // 오류 메시지를 확인할 수 있도록 10초 대기 후 종료
      await new Promise(resolve => setTimeout(resolve, 10000));
      process.exit(1);
    }
    
    console.log('서버가 성공적으로 시작되었습니다.');
    console.log('프로그램이 실행 중입니다. 종료하려면 Ctrl+C를 누르세요.');
    
    // 프로세스 종료 이벤트 처리
    process.on('SIGINT', () => {
      console.log('\n프로그램을 종료합니다.');
      server.close(() => {
        console.log('서버가 정상적으로 종료되었습니다.');
        process.exit(0);
      });
    });
    
    // Windows에서 스크립트가 종료되지 않도록 하는 코드
    process.stdin.resume(); // stdin을 열어 프로세스가 종료되지 않도록 함
    
    // 명시적인 무한 대기 코드
    await new Promise(() => {
      // 이 Promise는 의도적으로 resolve를 호출하지 않음 (무한 대기)
    });
  } catch (error) {
    console.error('프로그램 실행 중 치명적 오류:', error.message);
    console.error('스택 트레이스:', error.stack);
    console.error('프로그램이 10초 후에 종료됩니다...');
    
    // 오류 메시지를 확인할 수 있도록 10초 대기 후 종료
    await new Promise(resolve => setTimeout(resolve, 10000));
    process.exit(1);
  }
})(); 