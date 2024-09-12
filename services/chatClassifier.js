const mongoose = require('mongoose');
const Chat = require('../schemas/chat');
const axios = require('axios');

async function classifyTopics(roomId) {
  try {
    const chats = await Chat.find({ room: roomId }).sort('-createdAt').limit(100);
    
    const chatPoints = chats.map(chat => ({
      id: chat._id.toString(),
      nickname: chat.nickname,
      createdAt: chat.createdAt.toISOString(),
      content: chat.content
    }));

    const beamWidth = 5;
    let beams = [{ threads: [], score: 0 }];

    for (const chat of chatPoints) {
      let newBeams = [];
      for (const beam of beams) {
        const threadInputs = beam.threads.map(thread => thread);

        const newMessage = chat;

        try {
          const response = await axios.post('http://localhost:5000/predict', {
            thread: threadInputs,
            new_message: newMessage
          });

          const prediction = response.data.prediction;

          const topK = [{ score: prediction, index: beam.threads.length }]
            .concat(beam.threads.map((_, index) => ({ score: prediction, index })))
            .sort((a, b) => b.score - a.score)
            .slice(0, beamWidth);

          for (const { score, index } of topK) {
            const newThreads = [...beam.threads];
            if (index === newThreads.length) {
              newThreads.push([chat]);
            } else {
              newThreads[index].push(chat);
            }
            newBeams.push({ threads: newThreads, score: beam.score + Math.log(score) });
          }
        } catch (error) {
          console.error('Error calling Python backend:', error);
          // 에러 발생 시 현재 빔을 그대로 유지
          newBeams.push(beam);
        }
      }
      beams = newBeams.sort((a, b) => b.score - a.score).slice(0, beamWidth);
    }

    const bestThreads = beams[0].threads;

    const result = { messages: {} };
    bestThreads.forEach((thread, index) => {
      result.messages[`topic${index + 1}`] = {
        chats: thread.map(chat => ({
          id: chat.id,
          nickname: chat.nickname,
          createdAt: chat.createdAt,
          content: chat.content
        }))
      };
    });

    return result;
  } catch (error) {
    console.error('Error in classifyTopics:', error);
    throw error;
  }
}

module.exports = classifyTopics;