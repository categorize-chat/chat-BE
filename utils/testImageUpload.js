const fs = require('fs');
const path = require('path');
const axios = require('axios');
const FormData = require('form-data');
const { S3Client, PutObjectCommand } = require('@aws-sdk/client-s3');
const dotenv = require('dotenv');
const crypto = require('crypto');

// 환경변수 로드
dotenv.config({ path: path.join(__dirname, '../.env') });

// 명령줄 인자 처리
const args = process.argv.slice(2);
if (args.length < 1) {
  console.error('사용법: node testImageUpload.js <이미지파일명> [userId] [JWT토큰]');
  console.error('예시: node testImageUpload.js test.jpg');
  console.error('예시: node testImageUpload.js test.jpg 유저아이디');
  console.error('예시: node testImageUpload.js test.jpg 유저아이디 JWT토큰');
  process.exit(1);
}

const imageName = args[0];
const userId = args[1] || 'test-user';
const jwtToken = args[2];

// 이미지 파일 경로
const imagePath = path.join(__dirname, '..', imageName);

// 파일 존재 확인
if (!fs.existsSync(imagePath)) {
  console.error(`오류: 파일 "${imageName}"을 찾을 수 없습니다.`);
  process.exit(1);
}

// 1. 직접 S3 업로드 함수
async function uploadDirectlyToS3() {
  try {
    // S3 클라이언트 초기화
    const s3Client = new S3Client({
      region: process.env.AWS_REGION,
      credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
      }
    });

    // 고유 파일명 생성
    const timestamp = Date.now();
    const randomString = crypto.randomBytes(8).toString('hex');
    const extension = imageName.split('.').pop();
    const fileName = `profile-${timestamp}-${randomString}.${extension}`;
    const key = `profiles/${userId}/${fileName}`;

    // 파일 읽기
    const fileContent = fs.readFileSync(imagePath);
    
    // 파일 타입 추측
    let contentType = 'image/jpeg'; // 기본값
    if (extension.toLowerCase() === 'png') contentType = 'image/png';
    if (extension.toLowerCase() === 'gif') contentType = 'image/gif';
    
    // 업로드 파라미터
    const uploadParams = {
      Bucket: process.env.AWS_S3_BUCKET_NAME,
      Key: key,
      Body: fileContent,
      ContentType: contentType,
      ACL: 'public-read'
    };
    
    // 업로드 명령 실행
    const command = new PutObjectCommand(uploadParams);
    await s3Client.send(command);
    
    // 이미지 URL 생성
    const imageUrl = `https://${process.env.AWS_S3_BUCKET_NAME}.s3.${process.env.AWS_REGION}.amazonaws.com/${key}`;
    
    console.log('✅ S3 직접 업로드 성공!');
    console.log(`📷 이미지 URL: ${imageUrl}`);
    
    return imageUrl;
  } catch (error) {
    console.error('❌ S3 업로드 오류:', error.message);
    throw error;
  }
}

// 2. API를 통한 업로드 함수
async function uploadViaAPI() {
  if (!jwtToken) {
    console.error('❌ API 업로드에는 JWT 토큰이 필요합니다.');
    return null;
  }
  
  try {
    // Form 데이터 생성
    const form = new FormData();
    form.append('profileImage', fs.createReadStream(imagePath));
    
    // API 요청
    const response = await axios.post(
      'http://localhost:8005/settings/profile-image',
      form,
      {
        headers: {
          ...form.getHeaders(),
          'Authorization': `Bearer ${jwtToken}`
        }
      }
    );
    
    console.log('✅ API 업로드 성공!');
    console.log('📷 응답:', JSON.stringify(response.data, null, 2));
    
    return response.data;
  } catch (error) {
    console.error('❌ API 업로드 오류:', error.message);
    if (error.response) {
      console.error('서버 응답:', error.response.data);
    }
    throw error;
  }
}

// 메인 실행 함수
async function main() {
  console.log(`🚀 이미지 "${imageName}" 업로드 테스트 시작`);
  
  try {
    // 1. S3 직접 업로드
    console.log('\n📤 S3에 직접 업로드 중...');
    const s3Url = await uploadDirectlyToS3();
    
    // 2. API를 통한 업로드 (JWT 토큰이 있는 경우)
    if (jwtToken) {
      console.log('\n📤 API를 통한 업로드 중...');
      await uploadViaAPI();
    } else {
      console.log('\n⚠️ JWT 토큰이 없어 API 업로드를 건너뜁니다.');
      console.log('JWT 토큰이 필요하다면 로그인 후 토큰을 얻어서 인자로 전달하세요.');
    }
    
    console.log('\n✨ 테스트 완료!');
  } catch (error) {
    console.error('\n❌ 테스트 실패:', error.message);
    process.exit(1);
  }
}

// 실행
main(); 