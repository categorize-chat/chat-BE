const passport = require('passport');
const kakao = require('./kakaoStrategy');

module.exports = () => {
  // serialize & deserialize는 JWT를 사용하므로 필요하지 않음
  
  // 카카오 전략 등록
  passport.use(kakao());
  
};