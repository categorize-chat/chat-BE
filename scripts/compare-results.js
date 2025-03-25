const fs = require('fs');
const path = require('path');

// 결과 파일 경로
const beforeLoadTestPath = path.join(__dirname, 'before_optimization_load_test.json');
const afterLoadTestPath = path.join(__dirname, 'after_optimization_load_test.json');
const beforeQueryMonitorPath = path.join(__dirname, 'before_optimization_query_monitor.json');
const afterQueryMonitorPath = path.join(__dirname, 'after_optimization_query_monitor.json');

// 파일 존재 여부 확인
function checkFilesExist() {
  const files = [
    { path: beforeLoadTestPath, name: '최적화 전 부하 테스트' },
    { path: afterLoadTestPath, name: '최적화 후 부하 테스트' },
    { path: beforeQueryMonitorPath, name: '최적화 전 쿼리 모니터링' },
    { path: afterQueryMonitorPath, name: '최적화 후 쿼리 모니터링' }
  ];
  
  const missingFiles = files.filter(file => !fs.existsSync(file.path));
  
  if (missingFiles.length > 0) {
    console.error('다음 결과 파일이 누락되었습니다:');
    missingFiles.forEach(file => console.error(`- ${file.name} (${file.path})`));
    console.error('\n먼저 README.md의 지침에 따라 모든 테스트를 실행하세요.');
    return false;
  }
  
  return true;
}

// 부하 테스트 결과 비교
function compareLoadTests() {
  try {
    const beforeData = JSON.parse(fs.readFileSync(beforeLoadTestPath, 'utf8'));
    const afterData = JSON.parse(fs.readFileSync(afterLoadTestPath, 'utf8'));
    
    const beforeSummary = beforeData.summary;
    const afterSummary = afterData.summary;
    
    console.log('\n===== 부하 테스트 결과 비교 =====');
    
    // 핵심 지표 비교
    const metrics = [
      { name: '처리량(msg/s)', before: parseFloat(beforeSummary.messagesPerSecond), after: parseFloat(afterSummary.messagesPerSecond) },
      { name: '평균 지연 시간(ms)', before: parseFloat(beforeSummary.avgLatency), after: parseFloat(afterSummary.avgLatency) },
      { name: '성공률(%)', before: parseFloat(beforeSummary.successRate), after: parseFloat(afterSummary.successRate) }
    ];
    
    const compareTable = metrics.map(metric => {
      const change = metric.after - metric.before;
      const percentChange = ((change / metric.before) * 100).toFixed(2);
      const improved = metric.name === '평균 지연 시간(ms)' ? change < 0 : change > 0;
      
      return {
        '지표': metric.name,
        '최적화 전': metric.before,
        '최적화 후': metric.after,
        '변화량': `${change > 0 ? '+' : ''}${change.toFixed(2)} (${change > 0 ? '+' : ''}${percentChange}%)`,
        '개선 여부': improved ? '✅ 개선' : '❌ 저하'
      };
    });
    
    console.table(compareTable);
  } catch (error) {
    console.error('부하 테스트 결과 비교 중 오류 발생:', error);
  }
}

// 쿼리 모니터링 결과 비교
function compareQueryMonitors() {
  try {
    const beforeData = JSON.parse(fs.readFileSync(beforeQueryMonitorPath, 'utf8'));
    const afterData = JSON.parse(fs.readFileSync(afterQueryMonitorPath, 'utf8'));
    
    const beforeSummary = beforeData.summary;
    const afterSummary = afterData.summary;
    
    console.log('\n===== MongoDB 쿼리 모니터링 결과 비교 =====');
    
    // 핵심 지표 비교
    const metrics = [
      { name: '총 쿼리 수', before: beforeSummary.totalQueries, after: afterSummary.totalQueries },
      { name: '초당 쿼리 수', before: parseFloat(beforeSummary.queriesPerSecond), after: parseFloat(afterSummary.queriesPerSecond) }
    ];
    
    const compareTable = metrics.map(metric => {
      const change = metric.after - metric.before;
      const percentChange = ((change / metric.before) * 100).toFixed(2);
      const improved = change < 0; // 쿼리 수는 적을수록 좋음
      
      return {
        '지표': metric.name,
        '최적화 전': metric.before,
        '최적화 후': metric.after,
        '변화량': `${change > 0 ? '+' : ''}${change.toFixed(2)} (${change > 0 ? '+' : ''}${percentChange}%)`,
        '개선 여부': improved ? '✅ 개선' : '❌ 저하'
      };
    });
    
    console.table(compareTable);
    
    // 컬렉션별 쿼리 비교
    console.log('\n===== 컬렉션별 쿼리 변화 =====');
    
    const beforeCollections = beforeData.details.collections;
    const afterCollections = afterData.details.collections;
    
    // 모든 컬렉션 이름 가져오기
    const allCollections = new Set([
      ...Object.keys(beforeCollections || {}),
      ...Object.keys(afterCollections || {})
    ]);
    
    const collectionsTable = Array.from(allCollections).map(collection => {
      const beforeCount = beforeCollections[collection] || 0;
      const afterCount = afterCollections[collection] || 0;
      const change = afterCount - beforeCount;
      const percentChange = beforeCount > 0 ? ((change / beforeCount) * 100).toFixed(2) : 'N/A';
      const improved = change < 0;
      
      return {
        '컬렉션': collection,
        '최적화 전': beforeCount,
        '최적화 후': afterCount,
        '변화량': `${change > 0 ? '+' : ''}${change} (${change > 0 ? '+' : ''}${percentChange}%)`,
        '개선 여부': improved ? '✅ 개선' : (change === 0 ? '⚠️ 변화 없음' : '❌ 저하')
      };
    }).sort((a, b) => b['최적화 전'] - a['최적화 전']); // 최적화 전 쿼리 수 기준 내림차순 정렬
    
    console.table(collectionsTable);
  } catch (error) {
    console.error('쿼리 모니터링 결과 비교 중 오류 발생:', error);
  }
}

// 최적화 효과 요약
function summarizeImprovements() {
  try {
    const beforeLoadData = JSON.parse(fs.readFileSync(beforeLoadTestPath, 'utf8')).summary;
    const afterLoadData = JSON.parse(fs.readFileSync(afterLoadTestPath, 'utf8')).summary;
    const beforeQueryData = JSON.parse(fs.readFileSync(beforeQueryMonitorPath, 'utf8')).summary;
    const afterQueryData = JSON.parse(fs.readFileSync(afterQueryMonitorPath, 'utf8')).summary;
    
    // 처리량 개선율
    const throughputBefore = parseFloat(beforeLoadData.messagesPerSecond);
    const throughputAfter = parseFloat(afterLoadData.messagesPerSecond);
    const throughputImprovement = ((throughputAfter - throughputBefore) / throughputBefore * 100).toFixed(2);
    
    // 지연 시간 개선율
    const latencyBefore = parseFloat(beforeLoadData.avgLatency);
    const latencyAfter = parseFloat(afterLoadData.avgLatency);
    const latencyImprovement = ((latencyBefore - latencyAfter) / latencyBefore * 100).toFixed(2);
    
    // 쿼리 수 감소율
    const queriesPerSecondBefore = parseFloat(beforeQueryData.queriesPerSecond);
    const queriesPerSecondAfter = parseFloat(afterQueryData.queriesPerSecond);
    const queryReduction = ((queriesPerSecondBefore - queriesPerSecondAfter) / queriesPerSecondBefore * 100).toFixed(2);
    
    console.log('\n===== 최적화 효과 요약 =====');
    console.log(`1. 처리량(Throughput): ${throughputImprovement}% ${throughputImprovement > 0 ? '향상' : '감소'}`);
    console.log(`2. 지연 시간(Latency): ${latencyImprovement}% ${latencyImprovement > 0 ? '감소' : '증가'}`);
    console.log(`3. 초당 DB 쿼리 수: ${queryReduction}% ${queryReduction > 0 ? '감소' : '증가'}`);
    
    // 종합 평가
    const improvements = [
      parseFloat(throughputImprovement) > 0,
      parseFloat(latencyImprovement) > 0,
      parseFloat(queryReduction) > 0
    ];
    
    const positiveCount = improvements.filter(Boolean).length;
    
    console.log('\n===== 종합 평가 =====');
    if (positiveCount === 3) {
      console.log('🌟 모든 지표에서 성능이 향상되었습니다! 최적화가 매우 성공적입니다.');
    } else if (positiveCount === 2) {
      console.log('✅ 대부분의 지표에서 성능이 향상되었습니다. 최적화가 효과적입니다.');
    } else if (positiveCount === 1) {
      console.log('⚠️ 일부 지표에서만 성능이 향상되었습니다. 추가 최적화가 필요할 수 있습니다.');
    } else {
      console.log('❌ 성능이 향상된 지표가 없습니다. 최적화 방법을 재검토해야 합니다.');
    }
  } catch (error) {
    console.error('최적화 효과 요약 중 오류 발생:', error);
  }
}

// 메인 함수
function main() {
  console.log('채팅 서버 최적화 전후 성능 비교');
  console.log('===============================');
  
  if (!checkFilesExist()) {
    return;
  }
  
  compareLoadTests();
  compareQueryMonitors();
  summarizeImprovements();
  
  console.log('\n분석이 완료되었습니다.');
}

// 실행
main(); 