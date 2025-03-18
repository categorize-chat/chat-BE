const multer = require('multer');

// 메모리 스토리지 설정 (파일을 메모리에 버퍼로 저장)
const storage = multer.memoryStorage();

/**
 * 이미지 파일 필터링 함수
 */
const fileFilter = (req, file, cb) => {
  // 이미지 파일만 허용
  if (file.mimetype.startsWith('image/')) {
    cb(null, true);
  } else {
    cb(new Error('이미지 파일만 업로드 가능합니다.'), false);
  }
};

// 업로드 설정
const upload = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: 5 * 1024 * 1024, // 5MB 크기 제한
  }
});

module.exports = upload; 