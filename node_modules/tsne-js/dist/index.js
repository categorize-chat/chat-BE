'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _slicedToArray = function () { function sliceIterator(arr, i) { var _arr = []; var _n = true; var _d = false; var _e = undefined; try { for (var _i = arr[Symbol.iterator](), _s; !(_n = (_s = _i.next()).done); _n = true) { _arr.push(_s.value); if (i && _arr.length === i) break; } } catch (err) { _d = true; _e = err; } finally { try { if (!_n && _i["return"]) _i["return"](); } finally { if (_d) throw _e; } } return _arr; } return function (arr, i) { if (Array.isArray(arr)) { return arr; } else if (Symbol.iterator in Object(arr)) { return sliceIterator(arr, i); } else { throw new TypeError("Invalid attempt to destructure non-iterable instance"); } }; }();

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

var _events = require('events');

var _ndarray = require('ndarray');

var _ndarray2 = _interopRequireDefault(_ndarray);

var _ndarrayOps = require('ndarray-ops');

var _ndarrayOps2 = _interopRequireDefault(_ndarrayOps);

var _ndarrayPack = require('ndarray-pack');

var _ndarrayPack2 = _interopRequireDefault(_ndarrayPack);

var _ndarrayUnpack = require('ndarray-unpack');

var _ndarrayUnpack2 = _interopRequireDefault(_ndarrayUnpack);

var _cwise = require('cwise');

var _cwise2 = _interopRequireDefault(_cwise);

var _randn = require('./randn');

var _randn2 = _interopRequireDefault(_randn);

var _pairwiseDistances = require('./pairwise-distances');

var _pairwiseDistances2 = _interopRequireDefault(_pairwiseDistances);

var _jointProbabilities = require('./joint-probabilities');

var _jointProbabilities2 = _interopRequireDefault(_jointProbabilities);

var _klDivergence = require('./kl-divergence');

var _klDivergence2 = _interopRequireDefault(_klDivergence);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _toConsumableArray(arr) { if (Array.isArray(arr)) { for (var i = 0, arr2 = Array(arr.length); i < arr.length; i++) { arr2[i] = arr[i]; } return arr2; } else { return Array.from(arr); } }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

function _possibleConstructorReturn(self, call) { if (!self) { throw new ReferenceError("this hasn't been initialised - super() hasn't been called"); } return call && (typeof call === "object" || typeof call === "function") ? call : self; }

function _inherits(subClass, superClass) { if (typeof superClass !== "function" && superClass !== null) { throw new TypeError("Super expression must either be null or a function, not " + typeof superClass); } subClass.prototype = Object.create(superClass && superClass.prototype, { constructor: { value: subClass, enumerable: false, writable: true, configurable: true } }); if (superClass) Object.setPrototypeOf ? Object.setPrototypeOf(subClass, superClass) : subClass.__proto__ = superClass; }

var TSNE = function (_EventEmitter) {
  _inherits(TSNE, _EventEmitter);

  function TSNE(config) {
    _classCallCheck(this, TSNE);

    var _this = _possibleConstructorReturn(this, Object.getPrototypeOf(TSNE).call(this));

    config = config || {};

    _this.dim = config.dim || 2;
    _this.perplexity = config.perplexity || 30.0;
    _this.earlyExaggeration = config.earlyExaggeration || 4.0;
    _this.learningRate = config.learningRate || 1000.0;
    _this.nIter = config.nIter || 1000;
    _this.metric = config.metric || 'euclidean';

    _this.barneshut = config.barneshut || false;

    _this.inputData = null;
    _this.outputEmbedding = null;
    return _this;
  }

  _createClass(TSNE, [{
    key: 'init',
    value: function init(opts) {
      opts = opts || {};

      var inputData = opts.data || [];
      var type = opts.type || 'dense';

      // format input data as ndarray
      if (type === 'dense') {

        this.inputData = (0, _ndarrayPack2.default)(inputData);
      } else if (type === 'sparse') {

        var shape = [];
        var size = 1;

        var _loop = function _loop(d) {
          var dimShape = Math.max.apply(null, inputData.map(function (coord) {
            return coord[d];
          })) + 1;
          shape.push(dimShape);
          size *= dimShape;
        };

        for (var d = 0; d < inputData[0].length; d++) {
          _loop(d);
        }
        this.inputData = (0, _ndarray2.default)(new Float64Array(size), shape);
        var _iteratorNormalCompletion = true;
        var _didIteratorError = false;
        var _iteratorError = undefined;

        try {
          for (var _iterator = inputData[Symbol.iterator](), _step; !(_iteratorNormalCompletion = (_step = _iterator.next()).done); _iteratorNormalCompletion = true) {
            var _inputData;

            var coord = _step.value;

            (_inputData = this.inputData).set.apply(_inputData, _toConsumableArray(coord).concat([1]));
          }
        } catch (err) {
          _didIteratorError = true;
          _iteratorError = err;
        } finally {
          try {
            if (!_iteratorNormalCompletion && _iterator.return) {
              _iterator.return();
            }
          } finally {
            if (_didIteratorError) {
              throw _iteratorError;
            }
          }
        }
      } else {
        throw new Error('input data type must be dense or sparse');
      }

      // random initialization of output embedding
      this.outputEmbedding = (0, _randn2.default)(this.inputData.shape[0], this.dim);
    }
  }, {
    key: 'run',
    value: function run() {
      // calculate pairwise distances
      this.emit('progressStatus', 'Calculating pairwise distances');
      this.distances = (0, _pairwiseDistances2.default)(this.inputData, this.metric);

      this.emit('progressStatus', 'Calculating joint probabilities');
      this.alpha = Math.max(this.dim - 1, 1);
      this.P = (0, _jointProbabilities2.default)(this.distances, this.perplexity);

      var error = Number.MAX_VALUE;
      var iter = 0;

      // early exaggeration
      this.emit('progressStatus', 'Early exaggeration with momentum 0.5');
      _ndarrayOps2.default.mulseq(this.P, this.earlyExaggeration);

      var _gradDesc2 = this._gradDesc(iter, 50, 0.5, 0.0, 0.0);

      var _gradDesc3 = _slicedToArray(_gradDesc2, 2);

      error = _gradDesc3[0];
      iter = _gradDesc3[1];

      this.emit('progressStatus', 'Early exaggeration with momentum 0.8');


      // final optimization

      var _gradDesc4 = this._gradDesc(iter + 1, 100, 0.8, 0.0, 0.0);

      var _gradDesc5 = _slicedToArray(_gradDesc4, 2);

      error = _gradDesc5[0];
      iter = _gradDesc5[1];
      this.emit('progressStatus', 'Final optimization with momentum 0.8');
      _ndarrayOps2.default.divseq(this.P, this.earlyExaggeration);

      var _gradDesc6 = this._gradDesc(iter + 1, this.nIter, 0.8, 1e-6, 1e-6);

      var _gradDesc7 = _slicedToArray(_gradDesc6, 2);

      error = _gradDesc7[0];
      iter = _gradDesc7[1];


      this.emit('progressStatus', 'Optimization end');
      return [error, iter];
    }
  }, {
    key: 'rerun',
    value: function rerun() {
      // random re-initialization of output embedding
      this.outputEmbedding = (0, _randn2.default)(this.inputData.shape[0], this.dim);

      // re-run with gradient descent

      var _run = this.run();

      var _run2 = _slicedToArray(_run, 2);

      var error = _run2[0];
      var iter = _run2[1];


      return [error, iter];
    }
  }, {
    key: 'getOutput',
    value: function getOutput() {
      return (0, _ndarrayUnpack2.default)(this.outputEmbedding);
    }
  }, {
    key: 'getOutputScaled',
    value: function getOutputScaled() {
      // scale output embedding to [-1, 1]
      var outputEmbeddingScaled = (0, _ndarray2.default)(new Float64Array(this.outputEmbedding.size), this.outputEmbedding.shape);
      var temp = (0, _ndarray2.default)(new Float64Array(this.outputEmbedding.shape[0]), [this.outputEmbedding.shape[0]]);

      for (var d = 0; d < this.outputEmbedding.shape[1]; d++) {
        var maxVal = _ndarrayOps2.default.sup(_ndarrayOps2.default.abs(temp, this.outputEmbedding.pick(null, d)));
        _ndarrayOps2.default.divs(outputEmbeddingScaled.pick(null, d), this.outputEmbedding.pick(null, d), maxVal);
      }

      return (0, _ndarrayUnpack2.default)(outputEmbeddingScaled);
    }
  }, {
    key: '_gradDesc',
    value: function _gradDesc(iter, nIter, momentum) {
      var minGradNorm = arguments.length <= 3 || arguments[3] === undefined ? 1e-6 : arguments[3];
      var minErrorDiff = arguments.length <= 4 || arguments[4] === undefined ? 1e-6 : arguments[4];

      var nIterWithoutProg = 30;

      // initialize updates array
      var update = (0, _ndarray2.default)(new Float64Array(this.outputEmbedding.size), this.outputEmbedding.shape);

      // initialize gains array
      var tempArray = new Float64Array(this.outputEmbedding.size);
      for (var _i = 0; _i < tempArray.length; _i++) {
        tempArray[_i] = 1.0;
      }
      var gains = (0, _ndarray2.default)(tempArray, this.outputEmbedding.shape);

      var error = Number.MAX_VALUE;
      var bestError = Number.MAX_VALUE;
      var bestIter = 0;

      var norm = (0, _cwise2.default)({
        args: ['array'],
        pre: function pre(a) {
          this.sum = 0.0;
        },
        body: function body(a) {
          this.sum += a * a;
        },
        post: function post(a) {
          return Math.sqrt(this.sum);
        }
      });

      var gainsUpdate = (0, _cwise2.default)({
        args: ['array', 'array', 'array'],
        body: function body(c_gains, c_update, c_grad) {
          if (c_update * c_grad >= 0) {
            c_gains += 0.05;
          } else {
            c_gains *= 0.95;
          }
          // set mininum gain 0.01
          c_gains = Math.max(c_gains, 0.01);
        }
      });

      var i = undefined;
      for (i = iter; i < nIter; i++) {
        var _divergenceKL = (0, _klDivergence2.default)(this.outputEmbedding, this.P, this.alpha);

        var _divergenceKL2 = _slicedToArray(_divergenceKL, 2);

        var cost = _divergenceKL2[0];
        var grad = _divergenceKL2[1];

        var errorDiff = Math.abs(cost - error);
        error = cost;
        var gradNorm = norm(grad);

        this.emit('progressIter', [i, error, gradNorm]);

        if (error < bestError) {
          bestError = error;
          bestIter = i;
        } else if (i - bestIter > nIterWithoutProg) {
          break;
        }

        if (minGradNorm >= gradNorm) break;
        if (minErrorDiff >= errorDiff) break;

        gainsUpdate(gains, update, grad);
        _ndarrayOps2.default.muleq(grad, gains);

        var temp = (0, _ndarray2.default)(new Float64Array(grad.size), grad.shape);
        _ndarrayOps2.default.muls(temp, grad, this.learningRate);
        _ndarrayOps2.default.mulseq(update, momentum);
        _ndarrayOps2.default.subeq(update, temp);
        _ndarrayOps2.default.addeq(this.outputEmbedding, update);

        this.emit('progressData', this.getOutputScaled());
      }

      return [error, i];
    }
  }]);

  return TSNE;
}(_events.EventEmitter);

exports.default = TSNE;
module.exports = exports['default'];