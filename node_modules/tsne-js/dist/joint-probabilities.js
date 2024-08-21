'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

exports.default = function (distances, perplexity) {
  var nSteps = 100;
  var nSamples = distances.shape[0];
  var P_cond = (0, _ndarray2.default)(new Float64Array(nSamples * nSamples), [nSamples, nSamples]);
  var P = (0, _ndarray2.default)(new Float64Array(nSamples * nSamples), [nSamples, nSamples]);

  var beta = undefined,
      betaMin = undefined,
      betaMax = Infinity;
  var betaSum = 0.0;

  var desired_entropy = Math.log(perplexity);
  var entropyDiff = undefined,
      entropy = undefined;
  var sum_Pi = undefined,
      sum_disti_Pi = undefined;

  for (var i = 0; i < nSamples; i++) {
    beta = 1.0;
    betaMin = -Infinity;
    betaMax = Infinity;

    for (var step = 0; step < nSteps; step++) {

      for (var j = 0; j < nSamples; j++) {
        P_cond.set(i, j, Math.exp(-distances.get(i, j) * beta));
      }

      P_cond.set(i, i, 0.0);
      sum_Pi = 0.0;
      for (var j = 0; j < nSamples; j++) {
        sum_Pi += P_cond.get(i, j);
      }
      if (sum_Pi == 0.0) sum_Pi = EPSILON_DBL;

      sum_disti_Pi = 0.0;
      for (var j = 0; j < nSamples; j++) {
        P_cond.set(i, j, P_cond.get(i, j) / sum_Pi);
        sum_disti_Pi += distances.get(i, j) * P_cond.get(i, j);
      }

      entropy = Math.log(sum_Pi) + beta * sum_disti_Pi;
      entropyDiff = entropy - desired_entropy;
      if (Math.abs(entropyDiff) <= PERPLEXITY_TOLERANCE) break;

      if (entropyDiff > 0.0) {
        betaMin = beta;
        if (betaMax == Infinity) {
          beta = beta * 2.0;
        } else {
          beta = (beta + betaMax) / 2.0;
        }
      } else {
        betaMax = beta;
        if (betaMin == -Infinity) {
          beta = beta / 2.0;
        } else {
          beta = (beta + betaMin) / 2.0;
        }
      }
    }

    betaSum += beta;
  }

  _ndarrayOps2.default.add(P, P_cond, P_cond.transpose(1, 0));
  var sum_P = Math.max(_ndarrayOps2.default.sum(P), MACHINE_EPSILON);
  _ndarrayOps2.default.divseq(P, sum_P);
  _ndarrayOps2.default.maxseq(P, MACHINE_EPSILON);
  return P;
};

var _ndarray = require('ndarray');

var _ndarray2 = _interopRequireDefault(_ndarray);

var _ndarrayOps = require('ndarray-ops');

var _ndarrayOps2 = _interopRequireDefault(_ndarrayOps);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

var EPSILON_DBL = 1e-7;
var MACHINE_EPSILON = Number.EPSILON || 2.220446049250313e-16;
var PERPLEXITY_TOLERANCE = 1e-5;

module.exports = exports['default'];