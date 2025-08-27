const { S3Client, PutObjectCommand, DeleteObjectCommand } = require('@aws-sdk/client-s3');
const crypto = require('crypto');

const s3Client = new S3Client({
  region: process.env.AWS_REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
  }
});

// 사용자의 개인정보 보호 및 보안 상의 안전을 위함
const generateUniqueFileName = () => {
  const timestamp = Date.now();
  const randomString = crypto.randomBytes(4).toString('hex');
  return `profile-${timestamp}-${randomString}`;
};

const uploadProfileImage = async (file, userId) => {
  try {
    const fileName = generateUniqueFileName();
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

const deleteProfileImage = async (imageUrl) => {
  try {
    if (!imageUrl || !imageUrl.includes(process.env.AWS_S3_BUCKET_NAME)) {
      return;
    }
    
    const key = imageUrl.split('.com/')[1];
    
    const deleteParams = {
      Bucket: process.env.AWS_S3_BUCKET_NAME,
      Key: key
    };
    
    const command = new DeleteObjectCommand(deleteParams);
    await s3Client.send(command);
    console.log(`S3에서 이미지 삭제 완료`);
  } catch (error) {
    console.error('S3 이미지 삭제 오류:', error);
  }
};

module.exports = {
  uploadProfileImage,
  deleteProfileImage
}; 