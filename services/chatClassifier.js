const mongoose = require('mongoose');
const Chat = require('../schemas/chat');
const axios = require('axios');

const classifyTopics = async (roomId, howmany = 100) => {
  console.log(`roomId: ${roomId} 채팅 ${howmany}개 분류를 시작합니다.`);
  try {
    const chats = await Chat.find({ room: roomId })
      .sort({ createdAt: -1, _id: -1 })
      .limit(howmany)
      .lean();

    chats.reverse();
    
    const input = {
      channelId: roomId.toString(),
      howmany: howmany,
      chats: chats.map(chat => ({
        id: chat._id.toString(),
        nickname: chat.nickname,
        content: chat.content,
        createdAt: chat.createdAt.toISOString()
      }))
    };

    const response = await axios.post('http://localhost:5000/predict', input);
    
    if (!response.data) {
      throw new Error('모델 서버의 응답이 없습니다.');
    }

    const {topics, summaries} = response.data;

    const result = {
      refChat: chats[0],
      howmany,
      topics,
      summaries
    };

    console.log(`분류를 완료했습니다.`);
    return result;

  } catch (error) {
    console.error('분류 중 오류 발생: ', error);
    throw error;
  }
};

module.exports = classifyTopics;
