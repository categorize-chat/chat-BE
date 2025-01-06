const axios = require('axios');
const jwt = require('jsonwebtoken');
const User = require('../schemas/user');
const { generateToken } = require('../utils/jwt');


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
        ...targetUser,
      },
    });

  } catch (error) {
    console.error(error);
    next(error);
  }
}
