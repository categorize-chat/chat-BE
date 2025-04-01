const mongoose = require('mongoose');
const Chat = require('../schemas/chat');
const axios = require('axios');
require('dotenv').config();

const MODEL_API_URL = process.env.MODEL_API_URL || 'http://localhost:5000';

const classifyTopics = async (roomId, howmany = 100) => {
  console.log(`roomId: ${roomId} 채팅 ${howmany}개 분류 시작`);
  try {
    // MongoDB에서 최근 채팅 데이터 howmany개 가져오기
    const chats = await Chat.find({ room: roomId })
      .sort({ createdAt: -1, _id: -1 })
      .limit(howmany)
      .lean();

    chats.reverse();
    
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

    // Python 모델 서버로 채팅 분류 요청
    const response = await axios.post(`${MODEL_API_URL}/predict`, modelInput);
    
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

    console.log(`roomId: ${roomId} 분류 완료`);
    return result

  } catch (error) {
    console.error('Error in classifyTopics:', error);
    throw error;
  }
};

module.exports = classifyTopics;
