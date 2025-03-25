# 성능 테스트 지침서

이 폴더에는 채팅 서버의 성능을 테스트하고 최적화 효과를 측정하기 위한 스크립트가 포함되어 있습니다.

## 준비사항

테스트를 실행하기 전에 다음 패키지를 설치해야 합니다:

```bash
# 설치 스크립트 실행
chmod +x setup-tests.sh
./setup-tests.sh
```

## 테스트 종류

### 1. 부하 테스트 (load-test.js)

이 테스트는 여러 가상 클라이언트를 생성하여 서버에 대한 부하를 시뮬레이션합니다.

```bash
# 부하 테스트 실행
node scripts/load-test.js
```

**주요 측정 항목:**
- 초당 메시지 처리량
- 메시지 지연 시간
- 성공률
- 서버 리소스 사용량

### 2. MongoDB 쿼리 모니터링 (monitor-queries.js)

이 테스트는 서버가 실행되는 동안 MongoDB에 대한 쿼리를 모니터링합니다.

```bash
# 쿼리 모니터링 실행
node scripts/monitor-queries.js
```

**주요 측정 항목:**
- 총 쿼리 수
- 초당 쿼리 수
- 컬렉션별 쿼리 분포
- 작업 유형별 쿼리 분포

## 최적화 전후 비교 테스트

코드 최적화의 효과를 측정하려면 다음 단계를 따르세요:

1. **원본 코드 백업**
   ```bash
   cp chat-BE/socket.js chat-BE/socket.js.bak
   cp chat-BE/schemas/index.js chat-BE/schemas/index.js.bak
   ```

2. **최적화 전 테스트 실행**
   ```bash
   # 부하 테스트
   node scripts/load-test.js
   # 결과 파일 이름 변경
   mv scripts/load_test_result_*.json scripts/before_optimization_load_test.json
   
   # 쿼리 모니터링
   node scripts/monitor-queries.js
   # 결과 파일 이름 변경
   mv scripts/query_monitor_result_*.json scripts/before_optimization_query_monitor.json
   ```

3. **최적화 적용** (이미 완료됨)

4. **최적화 후 테스트 실행**
   ```bash
   # 부하 테스트
   node scripts/load-test.js
   # 결과 파일 이름 변경
   mv scripts/load_test_result_*.json scripts/after_optimization_load_test.json
   
   # 쿼리 모니터링
   node scripts/monitor-queries.js
   # 결과 파일 이름 변경
   mv scripts/query_monitor_result_*.json scripts/after_optimization_query_monitor.json
   ```

5. **결과 비교**
   ```bash
   # 간단한 비교 스크립트 실행
   node scripts/compare-results.js
   ```

## 테스트 구성 조정

테스트 스크립트 상단의 `CONFIG` 객체를 수정하여 테스트 매개변수를 조정할 수 있습니다:

- **load-test.js**:
  - `CLIENTS`: 동시 접속 클라이언트 수
  - `MESSAGES_PER_CLIENT`: 각 클라이언트가 보낼 메시지 수
  - `MESSAGE_INTERVAL`: 메시지 전송 간격 (ms)
  - `TEST_DURATION`: 총 테스트 시간 (ms)

- **monitor-queries.js**:
  - `DURATION`: 모니터링 지속 시간 (ms)
  - `SAMPLING_INTERVAL`: 샘플링 간격 (ms)

## 주의사항

- 테스트는 개발 환경에서만 실행하세요.
- 테스트 중에는 서버에 실제 사용자가 접속하지 않도록 하세요.
- 테스트 결과는 환경에 따라 달라질 수 있습니다. 