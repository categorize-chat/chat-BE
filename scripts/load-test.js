const { io } = require('socket.io-client');
const { v4: uuidv4 } = require('uuid');
const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');
const axios = require('axios');

// 환경 변수 로드
dotenv.config({ path: path.join(__dirname, '../.env') });

// 테스트 구성
const CONFIG = {
  SERVER_URL: 'http://localhost:8005', // 고정 값으로 변경
  CLIENTS: 10,            // 클라이언트 수 줄임
  MESSAGES_PER_CLIENT: 5, // 메시지 수 줄임
  MESSAGE_INTERVAL: 100,  // 메시지 전송 간격 (ms)
  ROOM_ID: null,          // 테스트할 방 ID (자동으로 생성됨)
  TEST_DURATION: 30 * 1000, // 총 테스트 시간 (ms)
};

// 결과 저장 객체
const results = {
  startTime: null,
  endTime: null,
  totalMessages: 0,
  successfulMessages: 0,
  failedMessages: 0,
  messageLatencies: [],  // 메시지 지연 시간 (ms)
  errors: [],
  cpuUsage: [],
  memoryUsage: [],
};

// 테스트 사용자 생성
async function createTestUser(index) {
  // Date.now()의 마지막 4자리만 사용하여 닉네임 길이 제한
  const timestamp = Date.now().toString().slice(-4);
  const nickname = `test_${index}_${timestamp}`;
  const email = `test${index}_${Date.now()}@example.com`;
  
  try {
    // 테스트 사용자 생성 (실제 서버 API 경로로 수정)
    const response = await axios.post(`${CONFIG.SERVER_URL}/user/join`, {
      nickname,
      email,
      password: 'password123'
    });
    
    // 응답 구조 디버깅
    console.log(`사용자 ${nickname} 생성 응답:`, JSON.stringify(response.data, null, 2));
    
    // 토큰 확인
    const token = response.data.result && response.data.result.accessToken 
      ? response.data.result.accessToken 
      : response.data.accessToken;
      
    if (!token) {
      console.error(`사용자 ${nickname}의 토큰이 없습니다. 응답:`, response.data);
    }
    
    return {
      id: response.data.result && response.data.result.userId 
        ? response.data.result.userId 
        : response.data.userId,
      nickname,
      email,
      token
    };
  } catch (error) {
    console.error(`사용자 생성 오류: ${error.message}`);
    if (error.response) {
      console.error('오류 응답:', error.response.data);
    }
    return null;
  }
}

// 테스트 방 생성
async function createTestRoom(ownerToken) {
  try {
    console.log('채팅방 생성에 사용되는 토큰:', ownerToken);
    
    const response = await axios.post(`${CONFIG.SERVER_URL}/chat`, {
      channelName: `테스트방_${Date.now()}`,
      description: '부하 테스트용 방'
    }, {
      headers: { 
        Authorization: `Bearer ${ownerToken}` 
      }
    });
    
    console.log('채팅방 생성 응답:', JSON.stringify(response.data, null, 2));
    
    const channelId = response.data.result && response.data.result.channelId 
      ? response.data.result.channelId 
      : (response.data.result && response.data.result.roomId 
        ? response.data.result.roomId 
        : response.data.channelId || response.data.roomId);
    
    return channelId;
  } catch (error) {
    console.error(`방 생성 오류: ${error.message}`);
    if (error.response) {
      console.error('오류 응답:', error.response.data);
    }
    return null;
  }
}

// 클라이언트 소켓 생성
function createSocketClient(user) {
  console.log(`사용자 ${user.nickname}의 소켓 연결 토큰:`, user.token);
  
  const socket = io(`${CONFIG.SERVER_URL}/chat`, {
    path: '/socket.io',
    transports: ['websocket'],
    auth: {
      token: user.token
    }
  });
  
  socket.on('connect_error', (err) => {
    console.error(`사용자 ${user.nickname}의 소켓 연결 오류:`, err.message);
  });
  
  return socket;
}

// 시스템 리소스 모니터링
function startMonitoring() {
  const monitorInterval = setInterval(() => {
    const memUsage = process.memoryUsage();
    results.memoryUsage.push({
      timestamp: Date.now(),
      rss: memUsage.rss / 1024 / 1024, // MB
      heapTotal: memUsage.heapTotal / 1024 / 1024, // MB
      heapUsed: memUsage.heapUsed / 1024 / 1024 // MB
    });
    
    const cpuUsage = process.cpuUsage();
    results.cpuUsage.push({
      timestamp: Date.now(),
      user: cpuUsage.user / 1000, // ms
      system: cpuUsage.system / 1000 // ms
    });
  }, 1000);
  
  return monitorInterval;
}

// 결과 저장
function saveResults() {
  results.endTime = Date.now();
  const testDuration = results.endTime - results.startTime;
  
  // 평균 지연 시간 계산
  const avgLatency = results.messageLatencies.length > 0
    ? results.messageLatencies.reduce((sum, val) => sum + val, 0) / results.messageLatencies.length
    : 0;
  
  // 초당 메시지 수 계산
  const messagesPerSecond = testDuration > 0
    ? (results.successfulMessages / (testDuration / 1000)).toFixed(2)
    : 0;
  
  const summary = {
    testConfig: CONFIG,
    testDuration: testDuration / 1000, // 초
    totalMessages: results.totalMessages,
    successfulMessages: results.successfulMessages,
    failedMessages: results.failedMessages,
    successRate: ((results.successfulMessages / results.totalMessages) * 100).toFixed(2) + '%',
    avgLatency: avgLatency.toFixed(2) + 'ms',
    messagesPerSecond,
    errorCount: results.errors.length,
    timestamp: new Date().toISOString()
  };
  
  console.log('\n===== 테스트 결과 요약 =====');
  console.table(summary);
  
  // 결과를 파일로 저장
  const resultFileName = `load_test_result_${new Date().toISOString().replace(/:/g, '-')}.json`;
  fs.writeFileSync(
    path.join(__dirname, resultFileName),
    JSON.stringify({ summary, details: results }, null, 2)
  );
  
  console.log(`\n결과가 ${resultFileName}에 저장되었습니다.`);
}

// 메인 테스트 실행
async function runLoadTest() {
  console.log('채팅 서버 부하 테스트를 시작합니다...');
  
  // 테스트 사용자 생성
  console.log(`${CONFIG.CLIENTS}명의 테스트 사용자를 생성합니다...`);
  const users = [];
  for (let i = 0; i < CONFIG.CLIENTS; i++) {
    const user = await createTestUser(i);
    if (user) users.push(user);
    process.stdout.write(`.`);
  }
  console.log(`\n${users.length}명의 사용자가 생성되었습니다.`);
  
  if (users.length === 0) {
    console.error('테스트 사용자를 생성할 수 없습니다. 테스트를 중단합니다.');
    return;
  }
  
  // 테스트 방 생성
  console.log('테스트 채팅방을 생성합니다...');
  CONFIG.ROOM_ID = await createTestRoom(users[0].token);
  if (!CONFIG.ROOM_ID) {
    console.error('테스트 채팅방을 생성할 수 없습니다. 테스트를 중단합니다.');
    return;
  }
  console.log(`채팅방이 생성되었습니다: ${CONFIG.ROOM_ID}`);
  
  // 모든 사용자를 채팅방에 가입시킴
  console.log('모든 사용자를 채팅방에 초대합니다...');
  for (const user of users) {
    try {
      console.log(`사용자 ${user.nickname}의 구독 요청 토큰:`, user.token);
      
      const response = await axios.post(`${CONFIG.SERVER_URL}/subscribe/${CONFIG.ROOM_ID}`, {}, {
        headers: { 
          Authorization: `Bearer ${user.token}` 
        }
      });
      
      console.log(`사용자 ${user.nickname}의 구독 응답:`, JSON.stringify(response.data, null, 2));
    } catch (error) {
      console.error(`채팅방 가입 오류 (${user.nickname}): ${error.message}`);
      if (error.response) {
        console.error('오류 응답:', error.response.data);
      }
    }
  }
  
  // 소켓 클라이언트 생성
  console.log('소켓 클라이언트를 생성합니다...');
  const clients = [];
  for (const user of users) {
    const socket = createSocketClient(user);
    clients.push({ user, socket });
  }
  
  // 모니터링 시작
  const monitorInterval = startMonitoring();
  
  // 모든 클라이언트 연결
  console.log('모든 클라이언트를 연결합니다...');
  await Promise.all(clients.map(client => {
    return new Promise((resolve) => {
      client.socket.on('connect', () => {
        console.log(`사용자 ${client.user.nickname} 소켓 연결 성공!`);
        
        // 채팅방 입장 (배열 형태인지 확인)
        client.socket.emit('join', Array.isArray(CONFIG.ROOM_ID) ? CONFIG.ROOM_ID : [CONFIG.ROOM_ID]);
        client.socket.emit('view', CONFIG.ROOM_ID);
        resolve();
      });
      
      client.socket.on('error', (err) => {
        console.error(`사용자 ${client.user.nickname}의 소켓 오류:`, err);
        results.errors.push({
          userId: client.user.id,
          error: err,
          timestamp: Date.now()
        });
      });
      
      client.socket.on('chat', (msg) => {
        // 메시지 수신 이벤트 처리
        // 필요하면 여기서 추가 로직
      });
    });
  }));
  
  console.log('테스트를 시작합니다...');
  results.startTime = Date.now();
  results.totalMessages = CONFIG.CLIENTS * CONFIG.MESSAGES_PER_CLIENT;
  
  // 각 클라이언트에서 메시지 전송
  for (const client of clients) {
    for (let i = 0; i < CONFIG.MESSAGES_PER_CLIENT; i++) {
      setTimeout(() => {
        const startTime = Date.now();
        const messageId = uuidv4();
        
        try {
          client.socket.emit('message', {
            room: CONFIG.ROOM_ID,
            content: `테스트 메시지 #${i} from ${client.user.nickname} [${messageId}]`
          });
          
          // 성공 횟수 증가
          results.successfulMessages++;
          
          // 지연 시간 기록
          const latency = Date.now() - startTime;
          results.messageLatencies.push(latency);
        } catch (err) {
          results.failedMessages++;
          results.errors.push({
            userId: client.user.id,
            messageId,
            error: err.message,
            timestamp: Date.now()
          });
        }
      }, i * CONFIG.MESSAGE_INTERVAL);
    }
  }
  
  // 테스트 완료 대기
  setTimeout(() => {
    // 모니터링 종료
    clearInterval(monitorInterval);
    
    // 모든 소켓 연결 종료
    for (const client of clients) {
      client.socket.disconnect();
    }
    
    // 결과 저장
    saveResults();
    
    console.log('테스트가 완료되었습니다.');
  }, CONFIG.TEST_DURATION);
}

// 테스트 실행
runLoadTest()
  .catch(err => {
    console.error('테스트 실행 중 오류 발생:', err);
    process.exit(1);
  }); 