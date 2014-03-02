'use strict';

var vector = require('./vector');
var matrix = require('./matrix');
var math = require('./math');

function oneMinus(num) {
  return 1 - num;
}

// Calculate cost and gradients.
//
// nnParams        - The unfolded weight matrices.
// inputLayerSize  - The number of units in the input layer.
// hiddenLayerSize - The number of units in the hidden layer.
// numLables       - The number of labels.
// X               - The Matrix of training data. m * featureCount
// y               - The Matrix of labels. m * 1
// lambda          - The regularization parameter???
//
// Returns an Array of cost and gradients.
function nnCostFunction(nnParams, inputLayerSize, hiddenLayerSize, numLabels, X, y, lambda) {
  var Theta1 = matrix.reshape(nnParams, hiddenLayerSize, inputLayerSize + 1);
  var Theta2 = matrix.reshape(nnParams, numLabels, hiddenLayerSize + 1, hiddenLayerSize * (inputLayerSize + 1));

  // The number of training data.
  var m = X.length;

  var J = 0;
  var Theta1Grad = matrix.make.apply(null, matrix.size(Theta1));
  var Theta2Grad = matrix.make.apply(null, matrix.size(Theta2));

  // Feed forward propagation.
  var a1 = matrix.hConcat(matrix.ones(m, 1), X);

  var z2 = matrix.applyMatrix(a1, matrix.transpose(Theta1));
  var a2 = matrix.hConcat(matrix.ones(m, 1), matrix.map(z2, math.sigmoid));

  var z3 = matrix.applyMatrix(a2, matrix.transpose(Theta2));
  var a3 = matrix.map(z3, math.sigmoid);

  // Calculate cost.
  var i, hk, yk;
  for (i = 0; i < numLabels; i++) {
    hk = matrix.pickColumn(a3, i);
    yk = y.map(function (label) {
      // label starts from 1 instead of 0.
      return label - 1 === i ? 1 : 0;
    });
    J += (
      - vector.product(yk, hk.map(Math.log))
      - vector.product(yk.map(oneMinus), hk.map(oneMinus).map(Math.log))
    ) / m;
  }

  // Regularize cost function.
  var T1 = matrix.flatten(matrix.pickColumns(Theta1, 1));
  var T2 = matrix.flatten(matrix.pickColumns(Theta2, 1));
  J += lambda / (2 * m) * (vector.product(T1, T1) + vector.product(T2, T2));

  // Calculate gradients.
  var gradients;

  return [J, gradients];
}

module.exports = nnCostFunction;
