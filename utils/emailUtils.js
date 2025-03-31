/**
 * 이메일 주소를 정규화하는 유틸리티 함수
 * 모든 이메일의 로컬 부분(@ 앞)의 점(.)을 제거하고,
 * 모든 이메일은 소문자로 변환합니다.
 */
function normalizeEmail(email) {
  if (!email) return email;
  
  // 이메일 주소를 소문자로 변환
  email = email.toLowerCase();
  
  // @ 기호를 기준으로 로컬 부분과 도메인 부분 분리
  const [local, domain] = email.split('@');
  
  // 모든 이메일의 로컬 부분에서 점(.) 제거
  const normalizedLocal = local.replace(/\./g, '');
  return `${normalizedLocal}@${domain}`;
}

module.exports = {
  normalizeEmail
}; 