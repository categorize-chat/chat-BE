'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

exports.default = function (embedding, P, alpha) {

  var nSamples = embedding.shape[0];
  var dim = embedding.shape[1];

  // Q: Student's t-distribution
  var Q = (0, _ndarray2.default)(new Float64Array(nSamples * nSamples), [nSamples, nSamples]);
  var n = (0, _pairwiseDistances2.default)(embedding, 'euclidean');
  square(n);
  var beta = (alpha + 1.0) / -2.0;
  _ndarrayOps2.default.powseq(_ndarrayOps2.default.divseq(_ndarrayOps2.default.addseq(n, 1), alpha), beta);
  for (var i = 0; i < nSamples; i++) {
    n.set(i, i, 0);
  }
  var sum_n = Math.max(_ndarrayOps2.default.sum(n), MACHINE_EPSILON);
  _ndarrayOps2.default.divs(Q, n, sum_n);
  _ndarrayOps2.default.maxseq(Q, MACHINE_EPSILON);

  // Kullback-Leibler divergence of P and Q
  var temp = (0, _ndarray2.default)(new Float64Array(nSamples * nSamples), [nSamples, nSamples]);
  var temp2 = (0, _ndarray2.default)(new Float64Array(nSamples * nSamples), [nSamples, nSamples]);
  _ndarrayOps2.default.div(temp, P, Q);
  _ndarrayOps2.default.logeq(temp);
  _ndarrayOps2.default.assign(temp2, P);
  var kl_divergence = _ndarrayOps2.default.sum(_ndarrayOps2.default.muleq(temp, temp2));

  // calculate gradient
  var grad = (0, _ndarray2.default)(new Float64Array(embedding.size), embedding.shape);
  var PQd = (0, _ndarray2.default)(new Float64Array(nSamples * nSamples), [nSamples, nSamples]);
  _ndarrayOps2.default.sub(PQd, P, Q);
  _ndarrayOps2.default.muleq(PQd, n);
  for (var i = 0; i < nSamples; i++) {
    for (var d = 0; d < dim; d++) {
      var _temp = (0, _ndarray2.default)(new Float64Array(embedding.shape[0]), [embedding.shape[0]]);
      _ndarrayOps2.default.assign(_temp, embedding.pick(null, d));
      _ndarrayOps2.default.addseq(_ndarrayOps2.default.negeq(_temp), embedding.get(i, d));
      _ndarrayOps2.default.muleq(_temp, PQd.pick(i, null));
      grad.set(i, d, _ndarrayOps2.default.sum(_temp));
    }
  }
  var c = 2.0 * (alpha + 1.0) / alpha;
  _ndarrayOps2.default.mulseq(grad, c);

  return [kl_divergence, grad];
};

var _ndarray = require('ndarray');

var _ndarray2 = _interopRequireDefault(_ndarray);

var _ndarrayOps = require('ndarray-ops');

var _ndarrayOps2 = _interopRequireDefault(_ndarrayOps);

var _cwise = require('cwise');

var _cwise2 = _interopRequireDefault(_cwise);

var _pairwiseDistances = require('./pairwise-distances');

var _pairwiseDistances2 = _interopRequireDefault(_pairwiseDistances);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

var MACHINE_EPSILON = Number.EPSILON || 2.220446049250313e-16;

var square = (0, _cwise2.default)({
  args: ['array'],
  body: function body(a) {
    a = a * a;
  }
});

module.exports = exports['default'];