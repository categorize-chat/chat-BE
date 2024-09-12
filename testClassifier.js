require('dotenv').config();
const mongoose = require('mongoose');
const axios = require('axios');
const { spawn } = require('child_process');
const classifyTopics = require('./services/chatClassifier');
const chatSchema = require('./schemas/chat');

mongoose.set('strictQuery', false);

async function connectToMongoDB() {
  try {
    await mongoose.connect(process.env.MONGODB_URI, { 
      dbName: 'aichat', 
      useNewUrlParser: true, 
      useUnifiedTopology: true 
    });
    console.log('MongoDB connected');
  } catch (err) {
    console.error('MongoDB connection error:', err);
    process.exit(1);
  }
}

let Chat;
try {
  Chat = mongoose.model('Chat');
} catch (error) {
  Chat = mongoose.model('Chat', chatSchema);
}

function trainModel() {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', ['model.py']);

    pythonProcess.stdout.on('data', (data) => {
      console.log(`Python stdout: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python stderr: ${data}`);
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        console.log('Model training completed successfully');
        resolve();
      } else {
        console.error(`Model training process exited with code ${code}`);
        reject(new Error(`Model training failed with code ${code}`));
      }
    });
  });
}

async function testClassifier(roomId) {
  const startTime = new Date();
  console.log(`Starting classification for room: ${roomId} at ${startTime.toISOString()}`);

  try {
    await connectToMongoDB();

    const objectIdRoomId = new mongoose.Types.ObjectId(roomId);

    const chats = await Chat.find({ room: objectIdRoomId }).sort('createdAt');
    console.log(`Found ${chats.length} chats for the room at ${new Date().toISOString()}`);

    if (chats.length === 0) {
      console.log('No chats found for this room. Please check the roomId.');
      return;
    }

    console.log('Sample chats:');
    chats.slice(0, 3).forEach(chat => {
      console.log(`${chat.nickname}: ${chat.content}`);
    });

    console.log('Starting model training...');
    await trainModel();
    console.log('Model training completed');

    console.log(`Starting classification at ${new Date().toISOString()}`);
    const result = await classifyTopics(objectIdRoomId);
    console.log(`Classification completed at ${new Date().toISOString()}`);

    console.log('Classification result:');
    if (Object.keys(result.messages).length === 0) {
      console.log('No topics were classified. This might indicate an issue with the classification logic.');
    } else {
      console.log(`Number of topics classified: ${Object.keys(result.messages).length}`);
      for (const [topic, data] of Object.entries(result.messages)) {
        console.log(`Topic: ${topic}`);
        console.log(`Number of chats: ${data.chats.length}`);
        console.log('Chats in this topic:');
        data.chats.forEach(chat => {
          console.log(`  ${chat.nickname}: ${chat.content}`);
        });
        console.log('---');
      }
    }
  } catch (error) {
    console.error('Error during classification:', error);
  } finally {
    await mongoose.disconnect();
    console.log(`Disconnected from MongoDB at ${new Date().toISOString()}`);
  }

  const endTime = new Date();
  console.log(`Classification process completed at ${endTime.toISOString()}`);
  console.log(`Total execution time: ${(endTime - startTime) / 1000} seconds`);
}

const testRoomId = '66b0fd658aab9f2bd7a41845'; // 실제 룸 ID
testClassifier(testRoomId);