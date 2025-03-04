const mongoose = require('mongoose');
const bcrypt = require('bcrypt');

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
    default: null
  },
  email: {
    type: String,
    unique: true,
    required: true,
  },
  password: {
    type: String,
  },
  isBanned: {
    type: Boolean,
    default: false
  },
  subscriptions: [{
    type: Schema.Types.ObjectId,
    ref: 'Room'
  }],
  isVerified: {
    type: Boolean,
    default: false
  },
  verificationToken: String,
  verificationTokenExpires: Date
}, {
  versionKey: false,
  id: false,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

userSchema.pre('save', async function(next) {
  if (this.password && this.isModified('password')) {
    try {
      const salt = await bcrypt.genSalt(10);
      this.password = await bcrypt.hash(this.password, salt);
      next();
    } catch (error) {
      next(error);
    }
  } else {
    next();
  }
});

userSchema.methods.comparePassword = async function(candidatePassword) {
  if (!this.password) return false;
  return await bcrypt.compare(candidatePassword, this.password);
};

userSchema.methods.generateVerificationToken = function() {
  const crypto = require('crypto');
  this.verificationToken = crypto.randomBytes(32).toString('hex');
  this.verificationTokenExpires = new Date(Date.now() + 24 * 60 * 60 * 1000); // 24시간
  return this.verificationToken;
};

module.exports = mongoose.model('User', userSchema);