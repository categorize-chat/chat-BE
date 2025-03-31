const redis = require('redis');
const { normalizeEmail } = require('./emailUtils');

// 상수 정의
const EXPIRY_TIME = 24 * 60 * 60; // Redis는 초 단위로 만료시간 설정
const USER_PREFIX = 'temp_user:';
const EMAIL_INDEX_KEY = 'email_to_token_index';

// Redis 클라이언트 생성 (싱글톤)
let clientInstance = null;

const getClient = async () => {
  if (!clientInstance) {
    clientInstance = redis.createClient({
      url: process.env.REDIS_URL || 'redis://localhost:6379'
    });
    
    clientInstance.on('error', (err) => console.error('Redis 연결 오류:', err));
    clientInstance.on('connect', () => console.log('Redis 서버에 연결됨'));
    
    await clientInstance.connect();
  }
  
  return clientInstance;
};

const tempStorage = {
  // 임시 사용자 저장
  saveTemp: async (token, userData) => {
    try {
      const client = await getClient();
      
      // 사용자 데이터 저장 (만료시간 설정)
      await client.setEx(
        `${USER_PREFIX}${token}`, 
        EXPIRY_TIME, 
        JSON.stringify(userData)
      );
      
      // 이메일 역방향 인덱스 추가 (이메일로 토큰 검색을 위해)
      if (userData.email) {
        // 이메일은 이미 정규화되어 있어야 함 (auth.js에서 처리)
        await client.hSet(EMAIL_INDEX_KEY, userData.email, token);
        // 이메일 인덱스도 같은 시간에 만료되도록 설정
        await client.expire(EMAIL_INDEX_KEY, EXPIRY_TIME);
      }
      
      return true;
    } catch (error) {
      console.error('Redis 저장 오류:', error);
      return false;
    }
  },

  // 임시 사용자 조회
  getTemp: async (token) => {
    try {
      const client = await getClient();
      const userData = await client.get(`${USER_PREFIX}${token}`);
      
      if (!userData) return null;
      return JSON.parse(userData);
    } catch (error) {
      console.error('Redis 조회 오류:', error);
      return null;
    }
  },

  // 임시 사용자 존재 여부 확인 (email 사용)
  getTokenByEmail: async (email) => {
    try {
      const client = await getClient();
      // 이메일은 이미 정규화되어 있어야 함 (auth.js에서 처리)
      
      // 이메일 인덱스에서 토큰 조회
      const token = await client.hGet(EMAIL_INDEX_KEY, email);
      
      if (!token) return null;
      
      // 토큰이 유효한지 확인
      const exists = await client.exists(`${USER_PREFIX}${token}`);
      return exists ? token : null;
    } catch (error) {
      console.error('Redis 이메일 검색 오류:', error);
      return null;
    }
  },

  // 임시 사용자 삭제
  removeTemp: async (token) => {
    try {
      const client = await getClient();
      // 사용자 데이터 가져오기
      const userData = await client.get(`${USER_PREFIX}${token}`);
      if (userData) {
        const parsed = JSON.parse(userData);
        if (parsed.email) {
          // 이메일 인덱스에서도 삭제
          await client.hDel(EMAIL_INDEX_KEY, parsed.email);
        }
      }
      
      // 사용자 데이터 삭제
      await client.del(`${USER_PREFIX}${token}`);
      return true;
    } catch (error) {
      console.error('Redis 삭제 오류:', error);
      return false;
    }
  }
};

module.exports = tempStorage;