const mongoose = require('mongoose');
const Chat = require('../schemas/chat');
const axios = require('axios');

const classifyTopics = async (roomId, howmany = 100) => {
  console.log(`roomId: ${roomId} 채팅 ${howmany}개 분류 시작`);
  try {
    const chats = await Chat.find({ room: roomId })
      .sort({ createdAt: -1, _id: -1 })
      .limit(howmany)
      .lean();

    chats.reverse();
    
    // content가 비어있는 채팅 필터링
    // sanitizeHtml을 사용하기 때문에 content가 비어있는 채팅이 생길 수 있음
    const validChats = chats.filter(chat => chat.content && chat.content.trim().length > 0);
    
    console.log(`전체 채팅: ${chats.length}개, 유효한 채팅: ${validChats.length}개`);
    
    if (validChats.length === 0) {
      return {
        refChat: null,
        howmany: 0,
        topics: [],
        summaries: []
      };
    }

    const input = {
      channelId: roomId.toString(),
      howmany: validChats.length,
      chats: validChats.map(chat => ({
        id: chat._id.toString(),
        nickname: (chat.user && chat.user.nickname) || chat.nickname || '알 수 없음',
        content: chat.content.trim(),
        createdAt: chat.createdAt.toISOString()
      }))
    };

    const response = await axios.post('http://localhost:5000/predict', input);
    
    if (!response.data) {
      throw new Error('모델 서버의 응답이 없습니다.');
    }

    const {topics, summaries} = response.data;

    const result = {
      refChat: validChats[0],
      howmany: validChats.length,
      topics,
      summaries
    };

    console.log(`분류 완료`);
    return result;

  } catch (error) {
    console.error('분류 중 오류 발생: ', error);
    throw error;
  }
};

module.exports = classifyTopics;
