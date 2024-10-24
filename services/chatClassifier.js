const mongoose = require('mongoose');
const Chat = require('../schemas/chat');
const axios = require('axios');

const classifyTopics = async (roomId, howmany = 100) => {
  console.log(`classifyTopics started for roomId: ${roomId} with limit: ${howmany}`);
  try {
    console.log('Fetching most recent chats from database...');
    // MongoDB에서 최근 채팅 데이터 N개 가져오기
    const chats = await Chat.find({ room: roomId })
      .sort({_id : -1})
      .limit(howmany)
      .lean();
    
    console.log(`Fetched ${chats.length} chats`);

    chats.reverse()
    
    // Python 모델에 보낼 데이터 형식으로 변환
    const modelInput = {
      channelId: roomId.toString(),
      howmany: howmany,
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

    // 응답이 모든 파라미터 세트 (low, mid, high)를 포함하는지 확인
    const expectedSets = ['low', 'mid', 'high'];
    for (const set of expectedSets) {
      if (!response.data.result[set]) {
        throw new Error(`Missing parameter set: ${set} in model response`);
      }
    }

    console.log('Classification completed for all parameter sets');
    return response.data.result;

  } catch (error) {
    console.error('Error in classifyTopics:', error);
    throw error;
  }
};

module.exports = classifyTopics;