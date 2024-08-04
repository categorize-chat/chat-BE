require('dotenv').config();
const mongoose = require('mongoose');
const { OpenAI } = require('openai');
const classifyTopics = require('./services/chatClassifier');
const chatSchema = require('./schemas/chat');

mongoose.set('strictQuery', false);

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

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

    try {
      const testEmbedding = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: "Test message",
      });
      console.log('OpenAI API test successful');
    } catch (apiError) {
      console.error('OpenAI API test failed:', apiError);
      return;
    }

    console.log(`Starting classification at ${new Date().toISOString()}`);
    const result = await classifyTopics(objectIdRoomId);
    console.log(`Classification completed at ${new Date().toISOString()}`);

    console.log('Classification result:');
    console.log(JSON.stringify(result, null, 2));

    if (Object.keys(result.messages).length === 0) {
      console.log('No topics were classified. This might indicate an issue with the classification logic.');
    } else {
      console.log(`Number of topics classified: ${Object.keys(result.messages).length}`);
      for (const [topic, data] of Object.entries(result.messages)) {
        console.log(`Topic: ${topic}`);
        console.log(`Number of chats: ${data.chats.length}`);
        console.log(`Summary: ${data.summary}`);
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

const testRoomId = '66a13a8bd4b4e6cca1cffc42'; // 실제 룸 ID
testClassifier(testRoomId);