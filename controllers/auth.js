const axios = require('axios');
const jwt = require('jsonwebtoken');
const User = require('../schemas/user');
const { generateToken } = require('../utils/jwt');
const bcrypt = require('bcrypt');
const { sendVerificationEmail } = require('../services/emailService');
const { generateVerificationToken } = require('../schemas/user');
const tempStorage = require('../utils/tempStorage');


exports.authKakao = async (req, res, next) => {
  try {
    const { code } = req.body;
    const kakaoTokenUrl = 'https://kauth.kakao.com/oauth/token';
  
    const kakaoToken = await axios.post(kakaoTokenUrl, new URLSearchParams({
      grant_type: 'authorization_code',
      client_id: process.env.KAKAO_CLIENT_ID,
      redirect_uri: process.env.KAKAO_REDIRECT_URI,
      code,
    }), {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded;charset=utf-8',
      }
    }).then(res => res.data);

    const kakaoUserUrl = 'https://kapi.kakao.com/v2/user/me';
    const user = await axios.get(kakaoUserUrl, {
      headers: {
        Authorization: `Bearer ${kakaoToken.access_token}`
      }
    }).then(res => res.data).then(data => ({
      nickname: data.properties.nickname,
      profileUrl: data.properties.profile_image,
      email: data.kakao_account.email,
    }));

    // 기존 유저 확인. email 을 키로.
    const exUser = await User.findOne({
      email: user.email,
    });

    // 기존 유저가 없다면 새로 생성
    const targetUser = exUser || await User.create({
      nickname: user.nickname,
      email: user.email,
      profileUrl: user.profileUrl,
    });

    

    // JWT 토큰 발급
    const {accessToken, refreshToken} = generateToken(targetUser);

    res.header(
      'Set-Cookie', `refreshToken=${refreshToken}; Path=/; HttpOnly`,
    );
    return res.json({
      isSuccess: true,
      code: 200,
      message: "요청에 성공했습니다.",
      result: {
        accessToken,
        nickname: targetUser.nickname,
        profileUrl: targetUser.profileUrl,
        email: targetUser.email
      },
    });

  } catch (error) {
    console.error(error);
    next(error);
  }
}

exports.loginUser = async (req, res, next) => {
  try {
    const { email, password } = req.body;

    // 이메일과 비밀번호 유효성 검사
    if (!email || !password) {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: "이메일과 비밀번호를 모두 입력해주세요."
      });
    }

    // Email로 사용자 찾기
    const user = await User.findOne({ email: email });
    
    // 사용자가 없거나 비밀번호가 일치하지 않는 경우
    if (!user || !(await user.comparePassword(password))) {
      return res.status(401).json({
        isSuccess: false,
        code: 401,
        message: "이메일 또는 비밀번호가 일치하지 않습니다."
      });
    }

    // 계정 정지 확인
    if (user.isBanned) {
      return res.status(403).json({
        isSuccess: false,
        code: 403,
        message: "계정이 정지되었습니다."
      });
    }

    // 이메일 인증 확인
    if (!user.isVerified) {
      return res.status(403).json({
        isSuccess: false,
        code: 403,
        message: "이메일 인증이 필요합니다. 이메일을 확인해주세요."
      });
    }

    // JWT 토큰 발급
    const { accessToken, refreshToken } = generateToken(user);

    // refreshToken을 HttpOnly 쿠키로 설정
    res.cookie('refreshToken', refreshToken, {
      path: '/',
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production', // 개발 환경에서는 false
    });

    // 응답 반환
    return res.json({
      isSuccess: true,
      code: 200,
      message: "요청에 성공했습니다.",
      result: {
        accessToken,
        nickname: user.nickname,
        email: user.email,
        profileUrl: user.profileUrl
      },
    });
  } catch (error) {
    console.error('Login error:', error);
    next(error);
  }
};

// chat-BE/controllers/auth.js
exports.registerLocalUser = async (req, res, next) => {
  try {
    console.log('Request body:', req.body);
    
    const { nickname, email, password } = req.body;

    // 필수 필드 확인
    if (!nickname || !email || !password) {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: "필수 정보가 누락되었습니다."
      });
    }

    // 이메일 중복 확인
    const existingUser = await User.findOne({ email: email });
    if (existingUser) {
      return res.status(409).json({
        isSuccess: false,
        code: 409,
        message: "이미 사용 중인 이메일입니다."
      });
    }

    // 닉네임 유효성 검사
    if (nickname.length < 2 || nickname.length > 20) {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: "닉네임은 2~20자 사이여야 합니다."
      });
    }

    // 비밀번호 유효성 검사 (간단한 예시)
    if (password.length < 8) {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: "비밀번호는 최소 8자 이상이어야 합니다."
      });
    }

    // 임시 사용자 객체 생성
    const userObj = {
      nickname,
      email,
      password,
      provider: 'local',
      profileUrl: "https://i.namu.wiki/i/UVVoIACG5XlxNksLitUb_U82uSi5vVlV7086nEtZfqXF0wNHBlpKJKMR9gBEekgUMZoSVr8NOl-JluZWy9De8q1dpwMg3ZMQuDR_GG7OdQXV49tS69czspC7FEP9vS3rC-cLIB6vEJ5oE0EBw_BN5g.webp"
    };

    // 인증 토큰 생성
    const crypto = require('crypto');
    const verificationToken = crypto.randomBytes(32).toString('hex');
    
    // 임시 저장소에 저장
    await tempStorage.saveTemp(verificationToken, userObj);

    // 인증 이메일 발송
    await sendVerificationEmail(email, verificationToken);

    // 응답
    return res.status(200).json({
      isSuccess: true,
      code: 200,
      message: "이메일을 확인하여 계정을 인증해주세요.",
      result: {
        email: email,
        nickname: nickname,
      }
    });
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({ error: error.message });
  }
};

exports.logoutUser = async (req, res, next) => {
  try {
    // 쿠키 삭제 (만료시간을 과거로 설정)
    res.cookie('refreshToken', '', {
      path: '/',
      expires: new Date(Date.now() - 3600000), // 현재 시간보다 1시간 전으로 설정
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production', // 개발 환경에서는 false, 배포 환경에서는 true
    });

    // 성공 응답 반환
    return res.status(200).json({
      isSuccess: true,
      code: 200,
      message: "요청에 성공했습니다",
      result: {}
    });
  } catch (error) {
    console.error('Logout error:', error);
    next(error);
  }
};

// 이메일 인증 처리 수정
exports.verifyEmail = async (req, res) => {
  try {
    const { token } = req.params;

    // 임시 저장소에서 사용자 정보 조회
    const userData = await tempStorage.getTemp(token);
    if (!userData) {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: "유효하지 않거나 만료된 인증 토큰입니다."
      });
    }

    // 실제 사용자 생성
    const newUser = await User.create({
      ...userData,
      isVerified: true
    });

    // 임시 데이터 삭제
    await tempStorage.removeTemp(token);

    return res.status(200).json({
      isSuccess: true,
      code: 200,
      message: "이메일 인증이 완료되었습니다. 이제 로그인할 수 있습니다."
    });
  } catch (error) {
    console.error('Email verification error:', error);
    res.status(500).json({ error: error.message });
  }
};

// 인증 이메일 재발송
exports.resendVerification = async (req, res) => {
  try {
    const { email } = req.body;

    // 이미 가입된 사용자인지 확인
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: "이미 가입된 이메일입니다."
      });
    }

    // 임시 저장소에서 이메일로 사용자 찾기
    let token = await tempStorage.getTokenByEmail(email);

    if (!token) {
      return res.status(404).json({
        isSuccess: false,
        code: 404,
        message: "가입 진행 중인 이메일을 찾을 수 없습니다."
      });
    }

    await sendVerificationEmail(email, token);

    return res.status(200).json({
      isSuccess: true,
      code: 200,
      message: "인증 이메일이 재발송되었습니다."
    });
  } catch (error) {
    console.error('Resend verification error:', error);
    res.status(500).json({ error: error.message });
  }
};
