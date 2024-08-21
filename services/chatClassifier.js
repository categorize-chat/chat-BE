const mongoose = require('mongoose');
const Chat = require('../schemas/chat');
const { OpenAI } = require('openai');
const { PCA } = require('ml-pca');

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

async function getEmbedding(chat) {
  if (chat.embedding) {
    return chat.embedding;
  }

  const response = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: chat.content,
  });
  
  const embedding = response.data[0].embedding;
  
  chat.embedding = embedding;
  await chat.save();
  
  return embedding;
}

function calculateTimeSimilarity(time1, time2) {
  const diffInMinutes = Math.abs(time1 - time2) / (1000 * 60);
  if (diffInMinutes <= 10) {
    return 1;
  } else {
    return Math.exp(-0.1 * (diffInMinutes - 10));
  }
}

function euclideanDistance(vec1, vec2) {
  return Math.sqrt(vec1.reduce((sum, v, i) => sum + Math.pow(v - vec2[i], 2), 0));
}

function normalizeVector(vec) {
  const magnitude = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0));
  return vec.map(v => v / magnitude);
}

function weightedSimilarity(point1, point2, speakerWeight = 0.2, timeWeight = 0.2) {
  const normalizedVec1 = normalizeVector(point1.reducedEmbedding);
  const normalizedVec2 = normalizeVector(point2.reducedEmbedding);
  const contentDistance = euclideanDistance(normalizedVec1, normalizedVec2);
  const contentSimilarity = 1 / (1 + contentDistance);
  const speakerSimilarity = point1.chat.nickname === point2.chat.nickname ? 1 : 0;
  const timeSimilarity = calculateTimeSimilarity(point1.chat.createdAt, point2.chat.createdAt);
  
  return contentSimilarity * (1 - speakerWeight - timeWeight) + 
         speakerSimilarity * speakerWeight + 
         timeSimilarity * timeWeight;
}

function findeps(points, k) {
  const distances = points.map(point => {
    const otherPoints = points.filter(p => p !== point);
    const kDistances = otherPoints
      .map(otherPoint => euclideanDistance(
        normalizeVector(point.reducedEmbedding),
        normalizeVector(otherPoint.reducedEmbedding)
      ))
      .sort((a, b) => a - b);
    return kDistances[k - 1];
  });

  distances.sort((a, b) => a - b);
  
  // 중앙값을 사용하여 eps 값 결정
  const medianIndex = Math.floor(distances.length / 2);
  return distances[medianIndex];
}

function dbscanWithSpeakerWeight(points, eps, minPts, speakerWeight, timeWeight) {
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
    embedding: await getEmbedding(chat),
    chat: chat
  })));

  // PCA를 사용한 차원 축소
  const embeddings = chatPoints.map(point => point.embedding);
  const pca = new PCA(embeddings);
  const reducedEmbeddings = pca.predict(embeddings, { nComponents: 10 });  // 10차원으로 축소

  const reducedChatPoints = chatPoints.map((point, index) => ({
    ...point,
    reducedEmbedding: reducedEmbeddings.getRow(index)
  }));

  const eps = 0.6;
  const minPoints = 2;
  const speakerWeight = 0.15;
  const timeWeight = 0.3;

  const { clusters, noise } = dbscanWithSpeakerWeight(reducedChatPoints, eps, minPoints, speakerWeight, timeWeight);

  const result = { messages: {} };

  for (const [index, cluster] of clusters.entries()) {
    const clusterChats = cluster.map(i => reducedChatPoints[i].chat);
    result.messages[`topic${index + 1}`] = {
      chats: clusterChats.map(chat => ({
        id: chat._id.toString(),
        nickname: chat.nickname,
        createdAt: chat.createdAt.toISOString(),
        content: chat.content
      }))
    };
  }

  if (noise.length > 0) {
    const noiseChats = noise.map(i => reducedChatPoints[i].chat);
    result.messages.uncategorized = {
      chats: noiseChats.map(chat => ({
        id: chat._id.toString(),
        nickname: chat.nickname,
        createdAt: chat.createdAt.toISOString(),
        content: chat.content
      }))
    };
  }

  return result;
}

module.exports = classifyTopics;