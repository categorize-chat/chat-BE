const mongoose = require('mongoose');

const {NODE_ENV, MONGODB_URI} = process.env;
const MONGO_URL = `${MONGODB_URI}`;

const connect = () => {
  if (NODE_ENV !== 'production') {
    mongoose.set('debug', true);
  }
  
  // 성능 관련 전역 설정
  mongoose.set('bufferCommands', NODE_ENV === 'production' ? false : true);

  mongoose.connect(MONGO_URL, {
    dbName: 'aichat',
    useNewUrlParser: true,
    maxPoolSize: 50, // 동시 연결 수 증가
    minPoolSize: 5, // 최소 풀 크기 유지
    socketTimeoutMS: 45000, // 소켓 타임아웃 설정
    keepAlive: true,
    keepAliveInitialDelay: 300000, // keepAlive 설정
    autoIndex: NODE_ENV !== 'production' // 프로덕션 환경에서는 자동 인덱싱 비활성화
  }).then(() => {
    console.log("몽고디비 연결 성공");
  }).catch((err) => {
    console.error("몽고디비 연결 에러", err);
  });
};

mongoose.connection.on('error', (error) => {
  console.error('몽고디비 연결 에러', error);
  if (error.name === 'MongoNetworkError') {
    console.log('네트워크 오류로 인한 연결 실패, 5초 후 재연결 시도');
    setTimeout(() => connect(), 5000); // 네트워크 오류 시 5초 후 재연결 시도
  }
});

mongoose.connection.on('disconnected', () => {
  console.error('몽고디비 연결이 끊겼습니다. 연결을 재시도합니다.');
  connect();
});

module.exports = connect;
