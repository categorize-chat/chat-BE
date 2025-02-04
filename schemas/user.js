const mongoose = require('mongoose');

const { Schema } = mongoose;
const userSchema = new Schema({
  nickname: {
    type: String,
    required: true,
  },
  userId: {
    type: String,
    default: function() {
      return this._id.toString();
    }
  },
  profileUrl: {
    type: String,
    required: true,
  },
  email: {
    type: String,
    unique: true,
    required: true,
  },
  isBanned: {
    type: Boolean,
    default: false
  },
  subscriptions: [{
    type: Schema.Types.ObjectId,
    ref: 'Room'
  }]
}, {
  versionKey: false,
  id: false,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

module.exports = mongoose.model('User', userSchema);