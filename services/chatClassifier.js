const mongoose = require('mongoose');
const Chat = require('../schemas/chat');
const axios = require('axios');

const classifyTopics = async (roomId) => {
  console.log(`classifyTopics started for roomId: ${roomId}`);
  try {
    // MongoDB에서 최근 채팅 데이터 100개 가져오기
    console.log('Fetching most recent chats from database...');
    const chats = await Chat.find({ room: roomId })
      .sort({ 
        createdAt: -1,  // 생성시간 역순으로 첫번째 정렬
        _id: -1         // 생성시간이 같을 경우 _id 역순으로 두번째 정렬
      })
      .limit(100)
      .lean();
    
    console.log(`Fetched ${chats.length} chats`);
    
    // 시간순으로 다시 정렬 (시간이 같을 경우 _id 순으로)
    chats.sort((a, b) => {
      const timeCompare = a.createdAt - b.createdAt;
      return timeCompare !== 0 ? timeCompare : a._id.toString().localeCompare(b._id.toString());
    });
    
    // Python 모델에 보낼 데이터 형식으로 변환
    const modelInput = {
      channelId: roomId.toString(),
      chats: chats.map(chat => ({
        id: chat._id.toString(),
        nickname: chat.nickname,
        content: chat.content,
        createdAt: chat.createdAt.toISOString()
      }))
    };

    // Python 모델 서버로 요청
    console.log('Sending request to Python model server');
    const response = await axios.post('http://localhost:5000/predict', modelInput);
    
    if (!response.data || !response.data.result) {
      throw new Error('Invalid response from model server');
    }

    console.log('Classification completed');
    return response.data.result;

  } catch (error) {
    console.error('Error in classifyTopics:', error);
    throw error;
  }
};

module.exports = classifyTopics;