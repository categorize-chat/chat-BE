'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

exports.default = function (data, metric) {
  var nSamples = data.shape[0];
  var distance = (0, _ndarray2.default)(new Float64Array(nSamples * nSamples), [nSamples, nSamples]);

  switch (metric) {
    case 'euclidean':
      for (var i = 0; i < nSamples; i++) {
        for (var j = i + 1; j < nSamples; j++) {
          var d = euclidean(data.pick(i, null), data.pick(j, null));
          distance.set(i, j, d);
          distance.set(j, i, d);
        }
      }
      break;
    case 'manhattan':
      for (var i = 0; i < nSamples; i++) {
        for (var j = i + 1; j < nSamples; j++) {
          var d = manhattan(data.pick(i, null), data.pick(j, null));
          distance.set(i, j, d);
          distance.set(j, i, d);
        }
      }
      break;
    case 'jaccard':
      for (var i = 0; i < nSamples; i++) {
        for (var j = i + 1; j < nSamples; j++) {
          var d = jaccard(data.pick(i, null), data.pick(j, null));
          distance.set(i, j, d);
          distance.set(j, i, d);
        }
      }
      break;
    case 'dice':
      for (var i = 0; i < nSamples; i++) {
        for (var j = i + 1; j < nSamples; j++) {
          var d = dice(data.pick(i, null), data.pick(j, null));
          distance.set(i, j, d);
          distance.set(j, i, d);
        }
      }
      break;
    default:
  }

  return distance;
};

var _ndarray = require('ndarray');

var _ndarray2 = _interopRequireDefault(_ndarray);

var _cwise = require('cwise');

var _cwise2 = _interopRequireDefault(_cwise);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

// Euclidean distance
var euclidean = (0, _cwise2.default)({
  args: ['array', 'array'],
  pre: function pre(a, b) {
    this.sum = 0.0;
  },
  body: function body(a, b) {
    var d = a - b;
    this.sum += d * d;
  },
  post: function post(a, b) {
    return Math.sqrt(this.sum);
  }
});

// Manhattan distance
var manhattan = (0, _cwise2.default)({
  args: ['array', 'array'],
  pre: function pre(a, b) {
    this.sum = 0.0;
  },
  body: function body(a, b) {
    this.sum += Math.abs(a - b);
  },
  post: function post(a, b) {
    return this.sum;
  }
});

// Jaccard dissimilarity
var jaccard = (0, _cwise2.default)({
  args: ['array', 'array'],
  pre: function pre(a, b) {
    this.tf = 0.0;
    this.tt = 0.0;
  },
  body: function body(a, b) {
    var a_bool = Math.round(a);
    var b_bool = Math.round(b);
    this.tf += +(a_bool != b_bool);
    this.tt += +(a_bool == 1 && b_bool == 1);
  },
  post: function post(a, b) {
    if (this.tf + this.tt === 0) return 1.0;
    return this.tf / (this.tf + this.tt);
  }
});

// Dice dissimilarity
var dice = (0, _cwise2.default)({
  args: ['array', 'array'],
  pre: function pre(a, b) {
    this.tf = 0.0;
    this.tt = 0.0;
  },
  body: function body(a, b) {
    var a_bool = Math.round(a);
    var b_bool = Math.round(b);
    this.tf += +(a_bool != b_bool);
    this.tt += +(a_bool == 1 && b_bool == 1);
  },
  post: function post(a, b) {
    if (this.tf + this.tt === 0) return 1.0;
    return this.tf / (this.tf + 2 * this.tt);
  }
});

module.exports = exports['default'];