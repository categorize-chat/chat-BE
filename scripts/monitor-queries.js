const mongoose = require('mongoose');
const dotenv = require('dotenv');
const path = require('path');
const fs = require('fs');
const http = require('http');

// 환경 변수 로드
dotenv.config({ path: path.join(__dirname, '../.env') });

// 모니터링 설정
const CONFIG = {
  DURATION: 60000, // 모니터링 지속 시간 (ms)
  SAMPLING_INTERVAL: 1000, // 샘플링 간격 (ms)
  CAPTURE_METHOD: 'proxy', // 'direct' 또는 'proxy'
};

// 결과 저장 객체
const results = {
  startTime: null,
  endTime: null,
  samples: [],
  totalQueries: 0,
  collections: {},
  operationTypes: {},
};

// 쿼리 카운터
let queryCounter = 0;
let currentSample = {
  timestamp: 0,
  queries: 0,
  operations: {},
  collections: {},
};

// 현재 샘플 초기화
function resetCurrentSample() {
  currentSample = {
    timestamp: Date.now(),
    queries: 0,
    operations: {},
    collections: {},
  };
}

// 모니터링 시작 (직접 연결 방식)
function startDirectMonitoring() {
  results.startTime = Date.now();
  
  // 새 샘플 초기화
  resetCurrentSample();
  
  // Mongoose 디버그 모드 활성화
  mongoose.set('debug', (collectionName, methodName, ...args) => {
    // 쿼리 카운터 증가
    queryCounter++;
    results.totalQueries++;
    
    // 현재 샘플에 추가
    currentSample.queries++;
    
    // 컬렉션별 통계
    if (!currentSample.collections[collectionName]) {
      currentSample.collections[collectionName] = 0;
    }
    currentSample.collections[collectionName]++;
    
    if (!results.collections[collectionName]) {
      results.collections[collectionName] = 0;
    }
    results.collections[collectionName]++;
    
    // 작업 유형별 통계
    if (!currentSample.operations[methodName]) {
      currentSample.operations[methodName] = 0;
    }
    currentSample.operations[methodName]++;
    
    if (!results.operationTypes[methodName]) {
      results.operationTypes[methodName] = 0;
    }
    results.operationTypes[methodName]++;
    
    // 원래 디버그 출력
    console.log(`[MongoDB] ${collectionName}.${methodName}()`);
  });
  
  // 샘플링 간격으로 데이터 수집
  const samplingInterval = setInterval(() => {
    // 현재 샘플을 결과에 추가
    results.samples.push({ ...currentSample });
    
    // 콘솔에 현재 상태 출력
    console.log(`쿼리/초: ${currentSample.queries} (총 쿼리: ${results.totalQueries})`);
    
    // 새 샘플 초기화
    resetCurrentSample();
  }, CONFIG.SAMPLING_INTERVAL);
  
  // 모니터링 종료 타이머
  setTimeout(() => {
    clearInterval(samplingInterval);
    results.endTime = Date.now();
    saveResults();
  }, CONFIG.DURATION);
  
  console.log(`MongoDB 쿼리 모니터링을 시작합니다. ${CONFIG.DURATION / 1000}초 동안 실행됩니다.`);
}

// HTTP 서버 모니터링
function startProxyMonitoring() {
  const PROXY_PORT = process.env.MONITOR_PORT || 9000;
  
  results.startTime = Date.now();
  resetCurrentSample();
  
  // 로컬 HTTP 서버 생성
  const server = http.createServer((req, res) => {
    if (req.method === 'POST' && req.url === '/mongo-query') {
      let body = '';
      
      req.on('data', chunk => {
        body += chunk.toString();
      });
      
      req.on('end', () => {
        try {
          const queryData = JSON.parse(body);
          
          // 쿼리 카운터 증가
          queryCounter++;
          results.totalQueries++;
          
          // 현재 샘플에 추가
          currentSample.queries++;
          
          // 컬렉션별 통계
          const collectionName = queryData.collection || 'unknown';
          if (!currentSample.collections[collectionName]) {
            currentSample.collections[collectionName] = 0;
          }
          currentSample.collections[collectionName]++;
          
          if (!results.collections[collectionName]) {
            results.collections[collectionName] = 0;
          }
          results.collections[collectionName]++;
          
          // 작업 유형별 통계
          const methodName = queryData.method || 'unknown';
          if (!currentSample.operations[methodName]) {
            currentSample.operations[methodName] = 0;
          }
          currentSample.operations[methodName]++;
          
          if (!results.operationTypes[methodName]) {
            results.operationTypes[methodName] = 0;
          }
          results.operationTypes[methodName]++;
          
          // 콘솔에 쿼리 로그 출력
          console.log(`[MongoDB] ${collectionName}.${methodName}()`);
          
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ success: true }));
        } catch (error) {
          console.error('쿼리 데이터 처리 오류:', error);
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ success: false, error: error.message }));
        }
      });
    } else {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ success: false, error: 'Not found' }));
    }
  });
  
  server.listen(PROXY_PORT, () => {
    console.log(`모니터링 프록시 서버가 ${PROXY_PORT} 포트에서 시작되었습니다.`);
    console.log('이제 schemas/index.js에 있는 mongoose.set("debug", ...) 부분을 수정하여');
    console.log('모니터링 HTTP 요청을 이 서버로 보내도록 해야 합니다.');
    console.log('\n다음 명령을 실행하여 schemas/index.js를 수정해주세요:');
    console.log('\nnode scripts/patch-mongoose-debug.js\n');
  });
  
  // 샘플링 간격으로 데이터 수집
  const samplingInterval = setInterval(() => {
    // 현재 샘플을 결과에 추가
    results.samples.push({ ...currentSample });
    
    // 콘솔에 현재 상태 출력
    console.log(`쿼리/초: ${currentSample.queries} (총 쿼리: ${results.totalQueries})`);
    
    // 새 샘플 초기화
    resetCurrentSample();
  }, CONFIG.SAMPLING_INTERVAL);
  
  // 모니터링 종료 타이머
  setTimeout(() => {
    clearInterval(samplingInterval);
    server.close();
    results.endTime = Date.now();
    saveResults();
  }, CONFIG.DURATION);
}

// 결과 저장
function saveResults() {
  const duration = (results.endTime - results.startTime) / 1000; // 초
  const queriesPerSecond = results.totalQueries / duration;
  
  const summary = {
    duration: `${duration.toFixed(2)}초`,
    totalQueries: results.totalQueries,
    queriesPerSecond: queriesPerSecond.toFixed(2),
    collectionsAccessed: Object.keys(results.collections).length,
    mostAccessedCollection: getMostAccessedItem(results.collections),
    mostCommonOperation: getMostAccessedItem(results.operationTypes),
    timestamp: new Date().toISOString()
  };
  
  console.log('\n===== MongoDB 쿼리 모니터링 결과 =====');
  console.table(summary);
  
  console.log('\n===== 컬렉션별 쿼리 수 =====');
  console.table(results.collections);
  
  console.log('\n===== 작업 유형별 쿼리 수 =====');
  console.table(results.operationTypes);
  
  // 결과를 파일로 저장
  const resultFileName = `query_monitor_result_${new Date().toISOString().replace(/:/g, '-')}.json`;
  fs.writeFileSync(
    path.join(__dirname, resultFileName),
    JSON.stringify({ summary, details: results }, null, 2)
  );
  
  console.log(`\n결과가 ${resultFileName}에 저장되었습니다.`);
  process.exit(0);
}

// 가장 많이 접근된 항목 찾기
function getMostAccessedItem(obj) {
  let maxItem = '';
  let maxCount = 0;
  
  for (const [key, count] of Object.entries(obj)) {
    if (count > maxCount) {
      maxCount = count;
      maxItem = key;
    }
  }
  
  return maxCount > 0 ? `${maxItem} (${maxCount}회)` : '없음';
}

// MongoDB 연결
async function connectToMongoDB() {
  const MONGO_URL = process.env.MONGODB_URI;
  
  try {
    await mongoose.connect(MONGO_URL, {
      dbName: 'aichat',
      useNewUrlParser: true,
    });
    console.log('MongoDB에 연결되었습니다. 모니터링을 시작합니다.');
    
    if (CONFIG.CAPTURE_METHOD === 'direct') {
      startDirectMonitoring();
    } else {
      startProxyMonitoring();
    }
  } catch (error) {
    console.error('MongoDB 연결 오류:', error);
    process.exit(1);
  }
}

// 명령줄 인자 처리
const args = process.argv.slice(2);
if (args.includes('--proxy') || args.includes('-p')) {
  CONFIG.CAPTURE_METHOD = 'proxy';
}

// 프로그램 실행
if (CONFIG.CAPTURE_METHOD === 'direct') {
  console.log('직접 연결 방식으로 모니터링을 시작합니다.');
  connectToMongoDB();
} else {
  console.log('프록시 방식으로 모니터링을 시작합니다.');
  startProxyMonitoring();
} 