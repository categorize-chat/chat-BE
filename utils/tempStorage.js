// 임시 사용자 저장소
const tempUsers = new Map();

// 24시간 후 자동 삭제
const EXPIRY_TIME = 24 * 60 * 60 * 1000;

const tempStorage = {
  // 임시 사용자 저장
  saveTemp: (token, userData) => {
    tempUsers.set(token, {
      data: userData,
      expires: Date.now() + EXPIRY_TIME
    });

    // 만료시간 후 자동 삭제
    setTimeout(() => {
      tempUsers.delete(token);
    }, EXPIRY_TIME);
  },

  // 임시 사용자 조회
  getTemp: (token) => {
    const userData = tempUsers.get(token);
    if (!userData) return null;
    
    if (Date.now() > userData.expires) {
      tempUsers.delete(token);
      return null;
    }
    
    return userData.data;
  },

  // 임시 사용자 존재 여부 확인 (email 사용)
  getTokenByEmail: (email) => {
    for (const [token, data] of tempUsers.entries()) {
      if (data.data.email === email) {
        return token;
      }
    }
    return null;
  },

  // 임시 사용자 삭제
  removeTemp: (token) => {
    tempUsers.delete(token);
  }
};

module.exports = tempStorage; 
