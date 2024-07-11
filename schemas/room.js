const mongoose = require('mongoose');

const { Schema } = mongoose;
const roomSchema = new Schema({
  channelName: {
    type: String,
    required: true,
  },
  channelId: {
    type: String,
    default: function() {
      return this._id.toString();
    }
  }
}, {
  versionKey: false,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

module.exports = mongoose.model('Room', roomSchema);