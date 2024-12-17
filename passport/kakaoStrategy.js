const KakaoStrategy = require('passport-kakao').Strategy;
const User = require('../schemas/user');
const { generateToken } = require('../utils/jwt');

module.exports = () => {
  return new KakaoStrategy({
    clientID: process.env.KAKAO_ID,
    callbackURL: 'http://localhost:8005/user/oauth/kakao',
  }, async (accessToken, refreshToken, profile, done) => {
    try {
      const exUser = await User.findOne({
        snsId: profile.id,
        provider: 'kakao',
      });

      if (exUser) {
        const tokens = generateToken(exUser);
        done(null, { user: exUser, tokens });
      } else {
        const newUser = await User.create({
          nickname: profile.username || profile._json?.properties?.nickname,
          snsId: profile.id,
          provider: 'kakao',
          profileImage: profile._json?.properties?.profile_image,
          email: profile._json?.kakao_account?.email,
        });
        const tokens = generateToken(newUser);
        done(null, { user: newUser, tokens });
      }
    } catch (error) {
      console.error(error);
      done(error);
    }
  });
};