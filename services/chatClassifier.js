const mongoose = require('mongoose');
const Chat = require('../schemas/chat');
const axios = require('axios');

const classifyTopics = async (roomId, howmany = 100, startMessageId) => {
  console.log(`roomId: ${roomId} 채팅 ${howmany}개 분류 시작${startMessageId ? ` (시작 메시지 ID: ${startMessageId})` : ''}`);
  try {
    // MongoDB 쿼리 조건 설정
    const query = { room: roomId };
    
    // startMessageId가 있는 경우, 해당 ID 이후의 메시지만 조회
    if (startMessageId) {
      // ObjectId를 사용하여 _id가 startMessageId보다 큰 문서 찾기
      query._id = { $gte: mongoose.Types.ObjectId(startMessageId) };
    }

    // MongoDB에서 채팅 데이터 가져오기
    const chats = await Chat.find(query)
      .sort({createdAt: 1, _id: 1})
      .limit(howmany)
      .lean();

    const modelInput = {
      channelId: roomId.toString(),
      howmany: howmany,
      chats: chats.map(chat => ({
        id: chat._id.toString(),
        user: chat.user.toString(),
        content: chat.content,
        createdAt: chat.createdAt.toISOString()
      }))
    };

    // Python 모델 서버로 채팅 분류 요청
    const response = await axios.post((process.env.AI_SERVER_URL || 'http://localhost:5000') + '/predict', modelInput);
    
    if (response.status === 500 || !response.data) {
      throw new Error('Invalid response from model server');
    }

    
    const {topics, summaries} = response.data

    const result = {
      refChat: chats[0],
      howmany,
      topics,
      summaries
    }

    console.log(`roomId: ${roomId} 분류 완료, ${chats.length}개 메시지 처리됨`);
    return result

  } catch (error) {
    console.error('Error in classifyTopics:', error);
    throw error;
  }
};

module.exports = classifyTopics;
