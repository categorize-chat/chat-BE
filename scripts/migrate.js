/**
 * 통합 마이그레이션 스크립트
 * 
 * 실행 방법:
 * - 모든 마이그레이션 실행: node scripts/migrate.js all
 * - readCounts 마이그레이션만 실행: node scripts/migrate.js readCounts
 * - lastMessage 마이그레이션만 실행: node scripts/migrate.js lastMessage
 */

const mongoose = require('mongoose');
const User = require('../schemas/user');
const Room = require('../schemas/room');
const Chat = require('../schemas/chat');
require('dotenv').config();

// MongoDB 연결
async function connectDB() {
  try {
    await mongoose.connect(process.env.MONGODB_URI, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    console.log('MongoDB 연결 성공');
  } catch (error) {
    console.error('MongoDB 연결 실패:', error);
    process.exit(1);
  }
}

// readCounts 마이그레이션 (배열에서 객체로)
async function migrateReadCounts() {
  console.log('\n===== readCounts 마이그레이션 시작 =====');
  try {
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

    console.log('readCounts 마이그레이션 완료');
    console.log(`성공: ${successCount}명, 실패: ${errorCount}명`);
    return { success: successCount, error: errorCount };
  } catch (error) {
    console.error('readCounts 마이그레이션 오류:', error);
    return { success: 0, error: 0 };
  }
}

// lastMessage 마이그레이션
async function migrateLastMessage() {
  console.log('\n===== lastMessage 마이그레이션 시작 =====');
  try {
    // 모든 채팅방 조회
    const rooms = await Room.find();
    console.log(`총 ${rooms.length}개의 채팅방을 처리합니다.`);
    
    let updatedCount = 0;
    let skippedCount = 0;
    
    // 각 채팅방에 대해 마지막 메시지 찾기
    for (const room of rooms) {
      // 해당 방의 마지막 메시지 찾기
      const lastChat = await Chat.findOne({ room: room._id })
        .sort({ createdAt: -1 })
        .limit(1);
      
      if (lastChat) {
        // 마지막 메시지가 있으면 lastMessage 필드 업데이트
        room.lastMessage = lastChat._id;
        await room.save();
        updatedCount++;
        console.log(`채팅방 ${room.channelName} (${room._id})의 마지막 메시지를 설정했습니다.`);
      } else {
        // 메시지가 없는 방은 건너뜀
        skippedCount++;
        console.log(`채팅방 ${room.channelName} (${room._id})에는 메시지가 없습니다.`);
      }
    }
    
    console.log('lastMessage 마이그레이션 완료');
    console.log(`총 ${rooms.length}개 채팅방 중 ${updatedCount}개 업데이트, ${skippedCount}개 건너뜀`);
    return { updated: updatedCount, skipped: skippedCount };
  } catch (error) {
    console.error('lastMessage 마이그레이션 오류:', error);
    return { updated: 0, skipped: 0 };
  }
}

// 메인 함수
async function main() {
  try {
    await connectDB();
    
    // 명령줄 인수 처리
    const args = process.argv.slice(2);
    const migrationType = args[0] || 'all';
    
    console.log(`마이그레이션 유형: ${migrationType}`);
    
    if (migrationType === 'all' || migrationType === 'readCounts') {
      await migrateReadCounts();
    }
    
    if (migrationType === 'all' || migrationType === 'lastMessage') {
      await migrateLastMessage();
    }
    
    console.log('\n===== 모든 마이그레이션 작업 완료 =====');
  } catch (error) {
    console.error('마이그레이션 오류:', error);
  } finally {
    // 연결 종료
    await mongoose.connection.close();
    console.log('MongoDB 연결 종료');
  }
}

// 스크립트 실행
main(); 
