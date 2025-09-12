// 원래 있던 verifyToken, verifyRefreshToken 미들웨어는 삭제됨
// 추후 개발 시 유의 필요

const jwt = require('jsonwebtoken');
const redis = require('redis');

const redisClient = redis.createClient({
  host: process.env.REDIS_HOST,
  port: process.env.REDIS_PORT
});

redisClient.on('error', (err) => {
  console.error('Redis 연결 오류:', err);
});

redisClient.on('connect', () => {
  console.log('Redis에 연결되었습니다.');
});

redisClient.connect().catch(err => {
  console.error('Redis 연결 실패:', err);
});

const generateToken = (user) => {
  const accessToken = jwt.sign(
    { 
      id: user._id,
      nickname: user.nickname,
      email: user.email,
      profileUrl: user.profileUrl
    },
    process.env.JWT_SECRET,
    { expiresIn: '1d' }
  );

  const refreshToken = jwt.sign(
    { id: user._id },
    process.env.JWT_REFRESH_SECRET,
    { expiresIn: '7d' }
  );

  return { accessToken, refreshToken };
};

const addRefreshTokenToBlacklist = async (refreshToken) => {
  try {
    if (!refreshToken) return;
    
    const decoded = jwt.decode(refreshToken);
    let expiry;
    if (decoded?.exp) {
      expiry = decoded.exp - Math.floor(Date.now() / 1000);
    } else {
      expiry = 604800;
    }
    
    if (expiry > 0) {
      await redisClient.setEx(`blacklist_refresh:${refreshToken}`, expiry, 'revoked');
    }
  } catch (error) {
    console.error('Refresh 토큰 블랙리스트 추가 오류:', error);
  }
};

const isRefreshTokenBlacklisted = async (refreshToken) => {
  try {
    if (!refreshToken) return false;
    const result = await redisClient.get(`blacklist_refresh:${refreshToken}`);
    return result !== null;
  } catch (error) {
    console.error('Refresh 토큰 블랙리스트 확인 오류:', error);
    return false;
  }
};

const revokeAllUserTokens = async (userId) => {
  try {
    await redisClient.setEx(`user_revoked:${userId}`, 604800, Date.now().toString()); // 7일간 저장
  } catch (error) {
    console.error('사용자 토큰 무효화 오류:', error);
  }
};

const isUserTokenRevoked = async (userId, tokenIssuedAt) => {
  try {
    const revokedTime = await redisClient.get(`user_revoked:${userId}`);
    if (!revokedTime) return false;
    
    return parseInt(revokedTime) > (tokenIssuedAt * 1000);
  } catch (error) {
    console.error('사용자 토큰 무효화 확인 오류:', error);
    return false;
  }
};

const cacheUserStatus = async (userId, isBanned, ttl = 300) => {
  try {
    await redisClient.setEx(`user_status:${userId}`, ttl, JSON.stringify({ isBanned }));
  } catch (error) {
    console.error('사용자 상태 캐싱 오류:', error);
  }
};

const getCachedUserStatus = async (userId) => {
  try {
    const cachedStatus = await redisClient.get(`user_status:${userId}`);
    return cachedStatus ? JSON.parse(cachedStatus) : null;
  } catch (error) {
    console.error('캐시된 사용자 상태 조회 오류:', error);
    return null;
  }
};

module.exports = {
  generateToken,
  addRefreshTokenToBlacklist,
  isRefreshTokenBlacklisted,
  revokeAllUserTokens,
  isUserTokenRevoked,
  cacheUserStatus,
  getCachedUserStatus
};
