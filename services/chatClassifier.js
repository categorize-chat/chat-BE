const mongoose = require('mongoose');
const { OpenAI } = require('openai');
const Chat = require('../schemas/chat');

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

async function getEmbedding(text) {
  const response = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
  });
  return response.data[0].embedding;
}

function cosineSimilarity(vec1, vec2) {
  const dotProduct = vec1.reduce((acc, val, i) => acc + val * vec2[i], 0);
  const mag1 = Math.sqrt(vec1.reduce((acc, val) => acc + val * val, 0));
  const mag2 = Math.sqrt(vec2.reduce((acc, val) => acc + val * val, 0));
  return dotProduct / (mag1 * mag2);
}

function weightedSimilarity(point1, point2, speakerWeight = 0.2, timeWeight = 0.1) {
  const contentSimilarity = cosineSimilarity(point1.embedding, point2.embedding);
  const speakerSimilarity = point1.chat.nickname === point2.chat.nickname ? 1 : 0;
  
  // 시간 차이에 따른 유사도 계산 (1시간 이내면 1, 그 이상이면 지수적으로 감소)
  const timeDiff = Math.abs(point1.chat.createdAt - point2.chat.createdAt) / (1000 * 60 * 60); // 시간 단위로 변환
  const timeSimilarity = Math.exp(-timeDiff); // 지수 감소 함수 사용

  return contentSimilarity * (1 - speakerWeight - timeWeight) + 
         speakerSimilarity * speakerWeight + 
         timeSimilarity * timeWeight;
}

function dbscanWithWeights(points, eps, minPts, speakerWeight, timeWeight) {
  const clusters = [];
  const visited = new Set();
  const noise = new Set();

  function expandCluster(pointIndex, neighbors) {
    const cluster = new Set([pointIndex]);
    for (let i = 0; i < neighbors.length; i++) {
      const neighborIndex = neighbors[i];
      if (!visited.has(neighborIndex)) {
        visited.add(neighborIndex);
        const neighborNeighbors = getNeighbors(neighborIndex);
        if (neighborNeighbors.length >= minPts) {
          neighbors.push(...neighborNeighbors);
        }
      }
      cluster.add(neighborIndex);
    }
    return Array.from(cluster);
  }

  function getNeighbors(pointIndex) {
    return points.reduce((neighbors, point, index) => {
      if (pointIndex !== index && weightedSimilarity(points[pointIndex], point, speakerWeight, timeWeight) >= eps) {
        neighbors.push(index);
      }
      return neighbors;
    }, []);
  }

  for (let i = 0; i < points.length; i++) {
    if (visited.has(i)) continue;
    visited.add(i);
    const neighbors = getNeighbors(i);
    if (neighbors.length < minPts) {
      noise.add(i);
    } else {
      const cluster = expandCluster(i, neighbors);
      clusters.push(cluster);
    }
  }

  return { clusters, noise: Array.from(noise) };
}

async function classifyTopics(roomId) {
  const chats = await Chat.find({ room: roomId }).sort('createdAt');
  const chatPoints = await Promise.all(chats.map(async (chat) => ({
    id: chat._id.toString(),
    embedding: await getEmbedding(chat.content),
    chat: chat
  })));

  const eps = 0.4; // 가중치를 고려하여 임계값 조정
  const minPoints = 2; // 최소 포인트 수
  const speakerWeight = 0.2; // 화자 가중치 (0 ~ 1 사이 값)
  const timeWeight = 0.15; // 시간 가중치 (0 ~ 1 사이 값)

  const { clusters, noise } = dbscanWithWeights(chatPoints, eps, minPoints, speakerWeight, timeWeight);

  const result = { messages: {} };

  // 클러스터 처리
  for (const [index, cluster] of clusters.entries()) {
    const clusterChats = cluster.map(i => chatPoints[i].chat);
    const clusterContent = clusterChats.map(chat => chat.content).join('\n');
    //const summary = await summarizeChats(clusterContent);

    result.messages[`topic${index + 1}`] = {
      chats: clusterChats.map(chat => ({
        id: chat._id.toString(),
        nickname: chat.nickname,
        createdAt: chat.createdAt.toISOString(),
        content: chat.content
      })),
      //summary: summary
    };
  }

  // 노이즈 처리
  if (noise.length > 0) {
    const noiseChats = noise.map(i => chatPoints[i].chat);
    const noiseContent = noiseChats.map(chat => chat.content).join('\n');
    //const noiseSummary = await summarizeChats(noiseContent);

    result.messages.uncategorized = {
      chats: noiseChats.map(chat => ({
        id: chat._id.toString(),
        nickname: chat.nickname,
        createdAt: chat.createdAt.toISOString(),
        content: chat.content
      })),
      //summary: noiseSummary
    };
  }

  return result;
}

/*
async function summarizeChats(content) {
  const response = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    messages: [
      { role: "system", content: "주어진 채팅 내용을 간략하게 요약해주세요. 주요 주제와 핵심 포인트만 언급해주세요." },
      { role: "user", content }
    ],
  });
  return response.choices[0].message.content;
}
*/

module.exports = classifyTopics;