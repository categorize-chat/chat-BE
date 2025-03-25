
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

module.exports = mongoose.model('Room', roomSchema);
