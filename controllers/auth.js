const axios = require('axios');
const jwt = require('jsonwebtoken');
const User = require('../schemas/user');
const { generateToken } = require('../utils/jwt');
const bcrypt = require('bcrypt');


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
    console.log('Request body:', req.body); // 요청 본문 로깅
    
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

    // 비밀번호 해싱은 User 모델의 pre save 훅에서 처리됨
    const newUser = new User({
      nickname,
      email,
      password, // 저장하기 전에 자동으로 해싱됩니다
      provider: 'local',
      profileUrl: "https://i.namu.wiki/i/UVVoIACG5XlxNksLitUb_U82uSi5vVlV7086nEtZfqXF0wNHBlpKJKMR9gBEekgUMZoSVr8NOl-JluZWy9De8q1dpwMg3ZMQuDR_GG7OdQXV49tS69czspC7FEP9vS3rC-cLIB6vEJ5oE0EBw_BN5g.webp"
    });

    await newUser.save();

    // JWT 토큰 생성
    const { accessToken, refreshToken } = generateToken(newUser);

    res.cookie('refreshToken', refreshToken, {
      path: '/',
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production', // 개발 환경에서는 false
    });


    // 응답
    return res.status(200).json({
      isSuccess: true,
      code: 200,
      message: "회원가입에 성공했습니다.",
      result: {
        accessToken,
        nickname: newUser.nickname,
        email: newUser.email,
        profileUrl: newUser.profileUrl
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