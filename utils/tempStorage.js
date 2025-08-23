const redis = require('redis');

const EXPIRATION_TIME = 24 * 60 * 60;
// 이메일로 토큰을 찾기 위한 룩업 테이블의 이름이라 생각하면 됨
const EMAIL_INDEX_KEY = 'tokenIndex';

let client = null;

const getClient = async () => {
  if (!client) {
    client = redis.createClient({
      url: process.env.REDIS_URL
    });
    
    client.on('error', (err) => console.error('Redis 서버 연결 오류:', err));
    client.on('connect', () => console.log('Redis 서버에 연결됨'));
    
    await client.connect();
  }
  
  return client;
};

const tempStorage = {
  saveTemp: async (token, userData) => {
    try {
      const tempClient = await getClient();
      
      // 만료 기간까지 동시에 설정해야 하므로 set 대신 setEx 사용함
      await tempClient.setEx(
        token,
        EXPIRATION_TIME, 
        JSON.stringify(userData)
      );
      
      if (userData.email) {
        await tempClient.hSet(EMAIL_INDEX_KEY, userData.email, token);
        await tempClient.expire(EMAIL_INDEX_KEY, EXPIRATION_TIME);
      }
      
      return true;
    } catch (error) {
      console.error('Redis 저장 오류:', error);
      return false;
    }
  },

  getTemp: async (token) => {
    try {
      const tempClient = await getClient();
      const userData = await tempClient.get(token);
      
      if (!userData) return null;
      return JSON.parse(userData);
    } catch (error) {
      console.error('Redis 조회 오류:', error);
      return null;
    }
  },

  getTokenByEmail: async (email) => {
    try {
      const tempClient = await getClient();

      const token = await tempClient.hGet(EMAIL_INDEX_KEY, email);
      
      if (!token) return null;
      
      const exists = await tempClient.exists(token);
      return exists ? token : null;
    } catch (error) {
      console.error('Redis 이메일 검색 오류:', error);
      return null;
    }
  },

  removeTemp: async (token) => {
    try {
      const tempClient = await getClient();
      const userData = await tempClient.get(token);
      if (userData) {
        const parsed = JSON.parse(userData);
        if (parsed.email) {
          await tempClient.hDel(EMAIL_INDEX_KEY, parsed.email);
        }
      }
      
      await tempClient.del(token);
      return true;
    } catch (error) {
      console.error('Redis 삭제 오류:', error);
      return false;
    }
  }
};

module.exports = tempStorage;