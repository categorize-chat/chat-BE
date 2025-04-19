const mongoose = require('mongoose');
const Chat = require('../schemas/chat');
const axios = require('axios');

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
      chats: chats.map(chat => {
        // 안전한 nickname 추출 로직
        let nickname;
        if (chat.nickname) {
          nickname = chat.nickname;
        } else if (chat.user && chat.user.nickname) {
          nickname = chat.user.nickname;
        } else if (chat.userId && typeof chat.userId === 'object' && chat.userId.nickname) {
          nickname = chat.userId.nickname;
        } else {
          // 기본값 설정 - 닉네임을 찾을 수 없는 경우
          nickname = '알 수 없음';
          console.warn(`닉네임을 찾을 수 없음: 채팅 ID ${chat._id}`);
        }
        
        return {
          id: chat._id.toString(),
          nickname: nickname,
          content: chat.content,
          createdAt: chat.createdAt.toISOString()
        };
      })
    };

    // Python 모델 서버로 채팅 분류 요청
    const response = await axios.post('http://localhost:5000/predict', modelInput);
    
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
