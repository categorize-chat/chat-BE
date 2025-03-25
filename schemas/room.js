const mongoose = require('mongoose');

const { Schema } = mongoose;
const roomSchema = new Schema({
  channelName: {
    type: String,
    required: true,
  },
  description: {
    type: String,
    default: '',
  },
  channelId: {
    type: String,
    default: function() {
      return this._id.toString();
    }
  },
  owner: {
    type: Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  participants: [{
    type: Schema.Types.ObjectId,
    ref: 'User'
  }],
  totalMessageCount: {
    type: Number,
    default: 0
  },
  lastMessage: {
    type: Schema.Types.ObjectId,
    ref: 'Chat'
  }
}, {
  versionKey: false,
  id: false,
  toJSON: { virtuals: true },
  toObject: { virtuals: true },
  timestamps: true
});

// 성능 최적화를 위한 인덱스 추가
roomSchema.index({ owner: 1 });
roomSchema.index({ participants: 1 });
roomSchema.index({ channelId: 1 }, { unique: true });

module.exports = mongoose.model('Room', roomSchema);
