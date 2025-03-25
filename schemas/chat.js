const mongoose = require('mongoose');
const User = require('./user');  // User 스키마 import

const { Schema } = mongoose;
const chatSchema = new Schema({
  room: {
    type: Schema.Types.ObjectId,
    required: true,
    ref: 'Room',
  },
  user: {
    type: Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  content: String,
  createdAt: {
    type: Date,
    default: Date.now,
  },
  topic: {
    type: Number,
    default: -1
  },
});

// 성능 최적화를 위한 인덱스 추가
chatSchema.index({ room: 1 });
chatSchema.index({ user: 1 });
chatSchema.index({ room: 1, createdAt: -1 });

module.exports = mongoose.model('Chat', chatSchema);