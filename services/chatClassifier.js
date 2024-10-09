const mongoose = require('mongoose');
const Chat = require('../schemas/chat');
const axios = require('axios');

const classifyTopics = async (roomId) => {
  console.log(`classifyTopics started for roomId: ${roomId}`);
  try {
    console.log('Fetching chats from database...');
    const chats = await Chat.find({ room: roomId }).sort('-createdAt').limit(100);
    console.log(`Fetched ${chats.length} chats`);
    
    const chatPoints = chats.map(chat => ({
      id: chat._id.toString(),
      nickname: chat.nickname,
      createdAt: chat.createdAt.toISOString(),
      content: chat.content
    }));

    const beamWidth = 5;
    let beams = [{ threads: [], score: 0 }];

    console.log('Starting classification process...');
    for (const [index, chat] of chatPoints.entries()) {
      console.log(`Processing chat ${index + 1}/${chatPoints.length}`);
      let newBeams = [];
      for (const beam of beams) {
        const threadInputs = beam.threads.map(thread => thread);

        const newMessage = {
          _id: chat.id,
          nickname: chat.nickname,
          createdAt: chat.createdAt,
          content: chat.content
        };

        try {
          console.log(`Sending prediction request to Python server for chat ${index + 1}`);
          console.log('New message:', newMessage);
          const startTime = Date.now();
          const response = await axios.post('http://localhost:5000/predict', {
            thread: threadInputs,
            new_message: newMessage
          });
          const endTime = Date.now();
          console.log(`Prediction received in ${endTime - startTime}ms`);

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
              // 중복 검사 추가
              if (!newThreads[index].some(msg => msg.id === chat.id)) {
                newThreads[index].push(chat);
              }
            }
            newBeams.push({ threads: newThreads, score: beam.score + Math.log(score) });
          }
        } catch (error) {
          console.error('Error calling Python backend:', error.message);
          newBeams.push(beam);
        }
      }
      beams = newBeams.sort((a, b) => b.score - a.score).slice(0, beamWidth);
    }

    console.log('Classification process completed');
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
    console.log(`Classification result: ${JSON.stringify(result, null, 2)}`);
    return result;
  } catch (error) {
    console.error('Error in classifyTopics:', error);
    throw error;
  }
};

module.exports = classifyTopics;
