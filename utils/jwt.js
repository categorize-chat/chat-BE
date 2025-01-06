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

const verifyToken = (token, isRefreshToken = false) => {
  try {
    const secret = isRefreshToken ? process.env.JWT_REFRESH_SECRET : process.env.JWT_SECRET;
    const decoded = jwt.verify(token, secret);
    return {
      valid: true,
      expired: false,
      decoded
    };
  } catch (error) {
    return {
      valid: false,
      expired: error.name === "TokenExpiredError",
      decoded: null
    };
  }
};

module.exports = {
  generateToken,
  verifyToken
};
