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
    type: User.schema,  // User 스키마를 직접 참조
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
  embedding: {
    type: [Number],
    default: null,
  },
});

module.exports = mongoose.model('Chat', chatSchema);