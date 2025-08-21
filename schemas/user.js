const mongoose = require('mongoose');
const bcrypt = require('bcrypt');
const crypto = require('crypto');

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
  lastProfileUpdateTime: {
    type: Date, // 마지막으로 업데이트한 바로 그 시간임을 유의해야 함
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
  readCounts: {
    type: Map,
    of: Number,
    default: {}
  },
  isVerified: {
    type: Boolean,
    default: false
  },
  verificationToken: String,
  verificationTokenExpires: Date,
  resetPasswordToken: String,
  resetPasswordExpires: Date
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
  this.verificationToken = crypto.randomBytes(32).toString('hex');
  this.verificationTokenExpires = new Date(Date.now() + 24 * 60 * 60 * 1000);
  return this.verificationToken;
};

userSchema.methods.generatePasswordResetToken = function() {
  this.resetPasswordToken = crypto.randomBytes(32).toString('hex');
  this.resetPasswordExpires = new Date(Date.now() + 60 * 60 * 1000);
  return this.resetPasswordToken;
};

module.exports = mongoose.model('User', userSchema);
