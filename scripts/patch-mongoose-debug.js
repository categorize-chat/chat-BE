const fs = require('fs');
const path = require('path');

const schemasIndexPath = path.join(__dirname, '../schemas/index.js');

// 원본 파일 백업
const backupPath = path.join(__dirname, '../schemas/index.js.bak');
if (!fs.existsSync(backupPath)) {
  console.log('원본 파일 백업 중...');
  fs.copyFileSync(schemasIndexPath, backupPath);
  console.log(`백업 완료: ${backupPath}`);
}

// 파일 내용 읽기
console.log('schemas/index.js 파일 수정 중...');
let content = fs.readFileSync(schemasIndexPath, 'utf8');

// mongoose.set('debug', true) 부분을 찾아 대체
const debugRegex = /mongoose\.set\(\s*['"]debug['"]\s*,\s*(true|false)\s*\)/;
const monitoringCode = `mongoose.set('debug', function(collectionName, methodName, ...args) {
  // 모니터링 서버로 쿼리 정보 전송
  try {
    const http = require('http');
    
    const data = JSON.stringify({
      collection: collectionName,
      method: methodName,
      args: args,
      timestamp: Date.now()
    });
    
    const options = {
      hostname: 'localhost',
      port: ${process.env.MONITOR_PORT || 9000},
      path: '/mongo-query',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(data)
      }
    };
    
    const req = http.request(options);
    req.write(data);
    req.end();
  } catch (error) {
    console.error('모니터링 서버 전송 오류:', error.message);
  }
  
  // 콘솔에도 출력
  console.log(\`[MongoDB] \${collectionName}.\${methodName}()\`);
})`;

// 기존 mongoose.set('debug', true) 대체
if (debugRegex.test(content)) {
  content = content.replace(debugRegex, monitoringCode);
} else {
  // 패턴을 찾지 못한 경우, connect 함수 시작 부분에 추가
  const connectRegex = /(const connect = \(\) => {)/;
  if (connectRegex.test(content)) {
    content = content.replace(connectRegex, `$1\n  ${monitoringCode};\n`);
  } else {
    console.error('파일에서 적절한 위치를 찾을 수 없습니다.');
    process.exit(1);
  }
}

// 파일 쓰기
fs.writeFileSync(schemasIndexPath, content);
console.log('schemas/index.js 파일이 수정되었습니다.');

console.log('\n===== 사용 방법 =====');
console.log('1. 모니터링 서버를 시작하세요:');
console.log('   node scripts/monitor-queries.js --proxy');
console.log('2. 서버를 재시작하세요:');
console.log('   npm start');
console.log('3. 부하 테스트를 실행하세요:');
console.log('   node scripts/load-test.js');
console.log('\n원본 파일로 복원하려면:');
console.log('node scripts/restore-mongoose-debug.js'); 