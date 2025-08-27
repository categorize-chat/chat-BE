const multer = require('multer');

const storage = multer.memoryStorage();

const fileFilter = (req, file, cb) => {
  // 이미지 파일만 허용해야 함
  if (file.mimetype.startsWith('image/')) {
    cb(null, true);
  } else {
    cb(new Error('이미지 파일만 업로드 가능합니다.'), false);
  }
};

const upload = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: 10 * 1024 * 1024, // 이미지 파일 10MB 크기 제한
    // 사이즈를 크게 잡은 이유는 gif 같은 대용량 파일을 사용할 수 있게 하기 위함
  }
});

module.exports = upload; 