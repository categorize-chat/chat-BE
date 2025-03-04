const { verifyToken } = require('../utils/jwt');
const User = require("../schemas/user");

const authMiddleware = async (req, res, next) => {
  const authHeader = req.headers.authorization;

  if (!authHeader?.startsWith('Bearer ')) {
    // refreshToken으로 새로운 accessToken 발급 시도
    const refreshToken = req.cookies.refreshToken;
    if (refreshToken) {
      try {
        const { valid, expired, decoded } = verifyToken(refreshToken, true);
        
        if (!valid) {
          return res.status(401).json({
            isSuccess: false,
            code: 401,
            message: expired ? '리프레시 토큰이 만료되었습니다. 다시 로그인해주세요.' : '유효하지 않은 토큰입니다.'
          });
        }

        // 사용자 정보 확인
        const user = await User.findById(decoded.id);
        if (!user) {
          return res.status(401).json({
            isSuccess: false,
            code: 401,
            message: '존재하지 않는 사용자입니다.'
          });
        }

        // 새로운 accessToken 발급
        const { generateToken } = require('../utils/jwt');
        const { accessToken } = generateToken(user);

        return res.json({
          isSuccess: true,
          code: 200,
          message: "토큰이 재발급되었습니다.",
          result: {
            accessToken: accessToken,
            nickname: user.nickname
          }
        });
      } catch (error) {
        console.error('Token refresh error:', error);
        return res.status(500).json({
          isSuccess: false,
          code: 500,
          message: '토큰 갱신 중 오류가 발생했습니다.'
        });
      }
    }

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

  req.user = decoded;
  
  // 제재 확인
  const user = await User.findById(decoded.id);
  if (user.isBanned) {
    return res.status(403).json({
      isSuccess: false,
      code: 403,
      message: "계정이 정지되었습니다."
    });
  }
  
  next();
};

module.exports = { authMiddleware };