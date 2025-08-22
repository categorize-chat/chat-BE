const jwt = require('jsonwebtoken');
const User = require("../schemas/user");
const { generateToken } = require('../utils/jwt');

const authMiddleware = async (req, res, next) => {
  const authHeader = req.headers.authorization;

  if (!authHeader?.startsWith('Bearer ')) {
    // refreshToken으로 새로운 accessToken 발급 시도
    const refreshToken = req.cookies.refreshToken;
    if (refreshToken) {
      try {
        const decoded = jwt.verify(refreshToken, process.env.JWT_REFRESH_SECRET);
        
        const user = await User.findById(decoded.id);
        if (!user) {
          return res.status(401).json({
            isSuccess: false,
            code: 401,
            message: '존재하지 않는 사용자입니다.'
          });
        }

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
        console.error('토큰 재발급 중 오류 발생: ', error);
        if (error.name === 'TokenExpiredError') {
          return res.status(401).json({
            isSuccess: false,
            code: 401,
            message: '리프레시 토큰이 만료되었습니다. 다시 로그인해주세요.'
          });
        }
        return res.status(401).json({
          isSuccess: false,
          code: 401,
          message: '유효하지 않은 리프레시 토큰입니다.'
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
  
  let decoded;
  try {
    decoded = jwt.verify(accessToken, process.env.JWT_SECRET);
    req.user = decoded;
  } catch (error) {
    if (error.name === 'TokenExpiredError') {
      return res.status(401).json({
        isSuccess: false,
        code: 401,
        message: '토큰이 만료되었습니다.'
      });
    }
    return res.status(401).json({
      isSuccess: false,
      code: 401,
      message: '유효하지 않은 토큰입니다.'
    });
  }
  
  // 계정 밴 확인용
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