const mongoose = require('mongoose');
const Chat = require('../schemas/chat');
const axios = require('axios');

const classifyTopics = async (roomId) => {
  console.log(`classifyTopics started for roomId: ${roomId}`);
  try {
    // MongoDB에서 채팅 데이터 가져오기
    console.log('Fetching chats from database...');
    const chats = await Chat.find({ room: roomId })
      .sort('createdAt')
      .limit(100)
      .lean();
    
    console.log(`Fetched ${chats.length} chats`);
    
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