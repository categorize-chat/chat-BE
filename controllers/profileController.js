const User = require('../schemas/user');
const { uploadProfileImage, deleteProfileImage } = require('../utils/s3Service');

/**
 * 프로필 이미지 업데이트 컨트롤러
 */
exports.updateProfileImage = async (req, res, next) => {
  try {
    // 요청에 파일이 없는 경우
    if (!req.file) {
      return res.status(400).json({
        isSuccess: false,
        code: 400,
        message: "이미지 파일을 업로드해주세요."
      });
    }

    const userId = req.user.id;
    
    // 사용자 찾기
    const user = await User.findById(userId);
    if (!user) {
      return res.status(404).json({
        isSuccess: false,
        code: 404,
        message: "사용자를 찾을 수 없습니다."
      });
    }
    
    // 기존 프로필 이미지 URL 저장
    const existingProfileUrl = user.profileUrl;
    
    // S3에 새 이미지 업로드
    const imageUrl = await uploadProfileImage(req.file, userId);
    
    // 사용자 프로필 URL 업데이트
    user.profileUrl = imageUrl;
    await user.save();
    
    // 기존 이미지가 S3에 있었다면 삭제 (기본 이미지는 제외)
    if (existingProfileUrl && !existingProfileUrl.includes('namu.wiki')) {
      await deleteProfileImage(existingProfileUrl);
    }

    // 소켓으로 프로필 업데이트 알림
    if (req.app.get('io')) {
      req.app.get('io').of('/chat').emit('profileUpdate', {
        userId: userId,
        profileUrl: imageUrl
      });
    }

    return res.status(200).json({
      isSuccess: true,
      code: 200,
      message: "프로필 이미지가 성공적으로 업데이트되었습니다.",
      result: {
        profileUrl: imageUrl
      }
    });
  } catch (error) {
    console.error('프로필 이미지 업데이트 오류:', error);
    return res.status(500).json({
      isSuccess: false,
      code: 500,
      message: error.message || "프로필 이미지 업데이트 중 오류가 발생했습니다."
    });
  }
}; 