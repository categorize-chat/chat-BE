const mongoose = require('mongoose');
const Room = require('./schemas/room');
const Chat = require('./schemas/chat');
require('dotenv').config();

async function migrateRooms() {
  try {
    await mongoose.connect(process.env.MONGODB_URI);
    console.log('MongoDB 연결 성공');
    
    const rooms = await Room.find({});
    console.log(`${rooms.length}개의 채팅방 마이그레이션 시작`);
    
    for (const room of rooms) {
      const chatCount = await Chat.countDocuments({ room: room._id });
      
      await Room.updateOne(
        { _id: room._id },
        { totalMessageCount: chatCount }
      );
      
      console.log(`채팅방 ${room._id}: 총 채팅 수 ${chatCount}개로 업데이트 완료`);
    }
    
    console.log('마이그레이션 완료');
    process.exit(0);
  } catch (error) {
    console.error('마이그레이션 오류:', error);
    process.exit(1);
  }
}

migrateRooms(); 