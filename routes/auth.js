const express = require('express');
const passport = require('passport');
const User = require('../schemas/user');
const { verifyToken } = require('../utils/jwt');
const router = express.Router();

const { authKakao } = require('../controllers/auth');
const authController = require('../controllers/auth');


// 카카오 로그인
// router.get('/kakao', passport.authenticate('kakao'));

// 카카오 콜백 처리
router.post('/kakao', authKakao);

// 토큰 갱신
router.post('/refresh', async (req, res) => {
  const refreshToken = req.cookies.refreshToken;
  
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

    // refreshToken을 HttpOnly 쿠키로 설정
    res.cookie('refreshToken', tokens.refreshToken, {
      path: '/',
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production', // 개발 환경에서는 false
    });

    res.json({
      isSuccess: true,
      code: 200,
      message: "토큰이 갱신되었습니다.",
      result: {
        accessToken: tokens.accessToken,
      }
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

// 이메일 인증
router.post('/verify/:token', authController.verifyEmail);

// 인증 이메일 재발송
router.post('/resend-verification', authController.resendVerification);

// 비밀번호 재설정 요청
router.post('/password-change-request', authController.requestPasswordReset);

// 비밀번호 재설정 토큰 확인
router.get('/password-change-request/:token', authController.verifyResetToken);

// 비밀번호 변경
router.post('/reset-password', authController.resetPassword);

module.exports = router;
