const mongoose = require('mongoose');
const dotenv = require('dotenv');
const path = require('path');
const fs = require('fs');

// 환경 변수 로드
dotenv.config({ path: path.join(__dirname, '../.env') });

// 모니터링 설정
const CONFIG = {
  DURATION: 60000, // 모니터링 지속 시간 (ms)
  SAMPLING_INTERVAL: 1000, // 샘플링 간격 (ms)
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

// 모니터링 시작
function startMonitoring() {
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

// 현재 샘플 초기화
function resetCurrentSample() {
  currentSample = {
    timestamp: Date.now(),
    queries: 0,
    operations: {},
    collections: {},
  };
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
  
  return `${maxItem} (${maxCount}회)`;
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
    startMonitoring();
  } catch (error) {
    console.error('MongoDB 연결 오류:', error);
    process.exit(1);
  }
}

// 프로그램 실행
connectToMongoDB(); 