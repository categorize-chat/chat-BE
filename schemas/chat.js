const mongoose = require('mongoose');

const { Schema } = mongoose;
const { Types: { ObjectId } } = Schema;
const chatSchema = new Schema({
  room: {
    type: ObjectId,
    required: true,
    ref: 'Room',
  },
  nickname: {
    type: String,
    required: true,
  },
  content: String,
  createdAt: {
    type: Date,
    default: Date.now,
  },
  topic: {
    type: String,
    default: null
  },
  embedding: {
    type: [Number],
    default: null,
  },
});

module.exports = mongoose.model('Chat', chatSchema);