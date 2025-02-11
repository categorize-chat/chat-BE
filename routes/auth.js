const express = require('express');
const passport = require('passport');
const User = require('../schemas/user');
const { verifyToken } = require('../utils/jwt');
const router = express.Router();

const { authKakao } = require('../controllers/auth');


// 카카오 로그인
// router.get('/kakao', passport.authenticate('kakao'));

// 카카오 콜백 처리
router.post('/kakao', authKakao);

// 토큰 갱신
router.post('/refresh', async (req, res) => {
  const { refreshToken } = req.body;
  
  if (!refreshToken) {
    return res.status(401).json({
      isSuccess: false,
      code: 401,
      message: 'Refresh 토큰이 필요합니다.'
    });
  }

  try {
    // refresh token 검증
    const { valid, expired, decoded } = verifyToken(refreshToken, true);

    if (!valid) {
      return res.status(401).json({
        isSuccess: false,
        code: 401,
        message: expired ? '만료된 토큰입니다.' : '유효하지 않은 토큰입니다.'
      });
    }

    // 사용자 정보 조회
    const user = await User.findById(decoded.id);
    if (!user) {
      return res.status(401).json({
        isSuccess: false,
        code: 401,
        message: '존재하지 않는 사용자입니다.'
      });
    }

    // 새로운 토큰 발급
    const { generateToken } = require('../utils/jwt');
    const tokens = generateToken(user);

    res.json({
      isSuccess: true,
      code: 200,
      message: "토큰이 갱신되었습니다.",
      result: tokens
    });
  } catch (error) {
    console.error('Token refresh error:', error);
    res.status(500).json({
      isSuccess: false,
      code: 500,
      message: '토큰 갱신 중 오류가 발생했습니다.'
    });
  }
});

// 로그인 상태 확인
router.get('/check', 
  passport.authenticate('jwt', { session: false }), 
  (req, res) => {
    res.json({
      isSuccess: true,
      code: 200,
      message: "인증된 사용자입니다.",
      result: {
        userId: req.user.id,
        nickname: req.user.nickname
      }
    });
  }
);

module.exports = router;
