const mongoose = require('mongoose');

const { Schema } = mongoose;
const roomSchema = new Schema({
  channelName: {
    type: String,
    required: true,
  },
});

module.exports = mongoose.model('Room', roomSchema);
