const { verifyToken } = require('../utils/jwt');

const authMiddleware = async (req, res, next) => {
  const authHeader = req.headers.authorization;

  if (!authHeader?.startsWith('Bearer ')) {
    return res.status(401).json({
      isSuccess: false,
      code: 401,
      message: '인증 토큰이 필요합니다.'
    });
  }

  const accessToken = authHeader.split(' ')[1];

  const { valid, expired, decoded } = verifyToken(accessToken);

  if (!valid) {
    return res.status(401).json({
      isSuccess: false,
      code: 401,
      message: expired ? '토큰이 만료되었습니다.' : '유효하지 않은 토큰입니다.'
    });
  }

  req.user = decoded;  // 검증된 사용자 정보를 req에 저장
  next();
};

module.exports = { authMiddleware };