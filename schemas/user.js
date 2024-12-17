const mongoose = require('mongoose');

const { Schema } = mongoose;
const userSchema = new Schema({
  nickname: {
    type: String,
    required: true,
  },
  userId: {
    unique: true,
    type: String,
    default: function() {
      return this._id.toString();
    }
  },
  snsId: {
    type: String,
    unique: true,
  },
  provider: {
    type: String,
    required: true,
  },
  profileImage: String,
  email: String
}, {
  versionKey: false,
  id: false,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

module.exports = mongoose.model('User', userSchema);