'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

exports.default = function (samples, dim) {

  var randArray = new Float64Array(samples * dim);

  for (var i = 0; i < randArray.length; i++) {
    randArray[i] = gaussRandom() * 1e-4;
  }

  return (0, _ndarray2.default)(randArray, [samples, dim]);
};

var _ndarray = require('ndarray');

var _ndarray2 = _interopRequireDefault(_ndarray);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

// random Gaussian distribution based on Box-Muller transform
function gaussRandom() {
  var u = 2 * Math.random() - 1;
  var v = 2 * Math.random() - 1;
  var r = u * u + v * v;
  if (r == 0 || r > 1) return gaussRandom();
  return u * Math.sqrt(-2 * Math.log(r) / r);
}

module.exports = exports['default'];