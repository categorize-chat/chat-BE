const passport = require('passport');
const kakao = require('./kakaoStrategy');

module.exports = () => {
  // serialize는 JWT를 사용하므로 필요하지 않음
  
  passport.use(kakao());
  
};