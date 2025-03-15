const { S3Client, PutObjectCommand, DeleteObjectCommand } = require('@aws-sdk/client-s3');
const crypto = require('crypto');

// S3 클라이언트 초기화
const s3Client = new S3Client({
  region: process.env.AWS_REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
  }
});

/**
 * 고유한 파일명 생성
 * @param {string} originalName - 원본 파일명
 * @returns {string} 생성된 고유 파일명
 */
const generateUniqueFileName = (originalName) => {
  const timestamp = Date.now();
  const randomString = crypto.randomBytes(8).toString('hex');
  const extension = originalName.split('.').pop();
  return `profile-${timestamp}-${randomString}.${extension}`;
};

/**
 * S3에 이미지 업로드
 * @param {Object} file - multer 업로드 파일 객체
 * @param {string} userId - 사용자 ID
 * @returns {Promise<string>} 업로드된 이미지 URL
 */
const uploadProfileImage = async (file, userId) => {
  try {
    const fileName = generateUniqueFileName(file.originalname);
    const key = `profiles/${userId}/${fileName}`;
    
    const uploadParams = {
      Bucket: process.env.AWS_S3_BUCKET_NAME,
      Key: key,
      Body: file.buffer,
      ContentType: file.mimetype,
      ACL: 'public-read'
    };
    
    const command = new PutObjectCommand(uploadParams);
    await s3Client.send(command);
    
    return `https://${process.env.AWS_S3_BUCKET_NAME}.s3.${process.env.AWS_REGION}.amazonaws.com/${key}`;
  } catch (error) {
    console.error('S3 이미지 업로드 오류:', error);
    throw new Error('이미지 업로드 중 오류가 발생했습니다.');
  }
};

/**
 * S3에서 이미지 삭제
 * @param {string} imageUrl - 삭제할 이미지 URL
 * @returns {Promise<void>}
 */
const deleteProfileImage = async (imageUrl) => {
  try {
    // 기본 이미지이거나 S3 URL이 아닌 경우 무시
    if (!imageUrl || !imageUrl.includes(process.env.AWS_S3_BUCKET_NAME)) {
      return;
    }
    
    // S3 URL에서 키 추출
    const key = imageUrl.split('.com/')[1];
    
    const deleteParams = {
      Bucket: process.env.AWS_S3_BUCKET_NAME,
      Key: key
    };
    
    const command = new DeleteObjectCommand(deleteParams);
    await s3Client.send(command);
    console.log(`S3에서 이미지 삭제 완료: ${key}`);
  } catch (error) {
    console.error('S3 이미지 삭제 오류:', error);
    // 삭제 실패해도 계속 진행 (중요하지 않은 작업)
  }
};

module.exports = {
  uploadProfileImage,
  deleteProfileImage
}; 