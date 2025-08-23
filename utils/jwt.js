// 원래 있던 verifyToken, verifyRefreshToken 미들웨어는 삭제됨
// 추후 개발 시 유의 필요

const jwt = require('jsonwebtoken');

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

module.exports = {
  generateToken
};
