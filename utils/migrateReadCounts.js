/**
 * readCounts 구조를 배열에서 객체로 마이그레이션하는 스크립트
 * 
 * 실행 방법: node utils/migrateReadCounts.js
 */

const mongoose = require('mongoose');
const User = require('../schemas/user');
require('dotenv').config();

async function migrateReadCounts() {
  try {
    // MongoDB 연결
    await mongoose.connect(process.env.MONGODB_URI, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    console.log('MongoDB 연결 성공');

    // 모든 사용자 조회
    const users = await User.find({});
    console.log(`총 ${users.length}명의 사용자 데이터를 마이그레이션합니다.`);

    let successCount = 0;
    let errorCount = 0;

    for (const user of users) {
      try {
        // 기존 배열 형태의 readCounts가 있는 경우
        if (Array.isArray(user.readCounts) && user.readCounts.length > 0) {
          // 새로운 객체 형태의 readCounts 생성
          const newReadCounts = {};
          
          // 배열의 각 항목을 객체로 변환
          for (const item of user.readCounts) {
            if (item.room && item.count !== undefined) {
              const roomId = item.room.toString();
              newReadCounts[roomId] = item.count;
            }
          }
          
          // 사용자 데이터 업데이트
          await User.updateOne(
            { _id: user._id },
            { $set: { readCounts: newReadCounts } }
          );
          
          successCount++;
          console.log(`사용자 ${user._id} 마이그레이션 성공`);
        } else {
          // readCounts가 없거나 비어있는 경우 빈 객체로 초기화
          await User.updateOne(
            { _id: user._id },
            { $set: { readCounts: {} } }
          );
          
          successCount++;
          console.log(`사용자 ${user._id} 빈 readCounts 초기화 성공`);
        }
      } catch (error) {
        errorCount++;
        console.error(`사용자 ${user._id} 마이그레이션 실패:`, error);
      }
    }

    console.log('마이그레이션 완료');
    console.log(`성공: ${successCount}명, 실패: ${errorCount}명`);
  } catch (error) {
    console.error('마이그레이션 오류:', error);
  } finally {
    // MongoDB 연결 종료
    await mongoose.connection.close();
    console.log('MongoDB 연결 종료');
  }
}

// 스크립트 실행
migrateReadCounts(); 
