'use strict';

var vector = require('./vector');
var matrix = require('./matrix');
var math = require('./math');

// Ported from http://msdn.microsoft.com/en-us/magazine/jj658979.aspx.

function NeuralNetwork(inputCount, hiddenCount, outputCount) {
  this.inputCount = inputCount;
  this.hiddenCount = hiddenCount;
  this.outputCount = outputCount;

  this.inputs = vector.make(inputCount);

  this.ihWeights = matrix.make(inputCount + 1, hiddenCount);

  this.ihOutputs = vector.make(hiddenCount);

  this.hoWeights = matrix.make(hiddenCount + 1, outputCount);

  this.outputs = vector.make(outputCount);

  this.ihPrevWeightsDelta = matrix.make(inputCount + 1, hiddenCount);
  this.hoPrevWeightsDelta = matrix.make(hiddenCount + 1, outputCount);
}

NeuralNetwork.prototype.computeOutputs = function (xValues) {
  var i, j;

  if (xValues.length !== this.inputCount) {
    throw new Error('Expected ' + this.inputCount + ' inputs but got ' + xValues.length);
  }

  vector.copy(xValues, this.inputs);

  // Compute input-to-hidden weighted sums.
  var ihSums = vector.applyMatrix([1.0].concat(this.inputs), this.ihWeights);
  this.ihOutputs = ihSums.map(math.sigmoid);

  // TODO: Why different function `hyperTanh`?
  var hoSums = vector.applyMatrix([1.0].concat(this.ihOutputs), this.hoWeights);
  this.outputs = hoSums.map(math.hyperTanh);

  var result = vector.make(this.outputCount);
  vector.copy(this.outputs, result);
  return result;
};

NeuralNetwork.prototype.updateWeights = function (tValues, eta, alpha) {
  var i, j;
  var derivative, sum, delta;
  var outputGradients = vector.make(this.outputCount);
  var hiddenGradients = vector.make(this.hiddenCount);

  if (tValues.length !== this.outputCount) {
    throw new Error('Target values not same length as output.');
  }

  // -- Calculate gradients.

  // Compute output gradients.
  for (i = 0; i < this.outputCount; i++) {
    derivative = (1 - this.outputs[i]) * (1 + this.outputs[i]);
    outputGradients[i] = derivative * (tValues[i] - this.outputs[i]);
  }

  // Compute hidden gradients.
  for (i = 0; i < this.hiddenCount; i++) {
    derivative = (1 - this.ihOutputs[i]) * this.ihOutputs[i];
    sum = 0.0;
    for (j = 0; j < this.outputCount; j++) {
      sum += outputGradients[j] * this.hoWeights[i + 1][j];
    }
    hiddenGradients[i] = derivative * sum;
  }

  // -- Update weights with gradients.

  // Update input-to-hidden weights.
  for (i = 0; i < this.inputCount + 1; i++) {
    for (j = 0; j < this.hiddenCount; j++) {
      delta = eta * hiddenGradients[j] * (i === 0 ? 1.0 : this.inputs[i - 1]);
      this.ihWeights[i][j] += delta + alpha * this.ihPrevWeightsDelta[i][j];
      this.ihPrevWeightsDelta[i][j] = delta;
    }
  }

  // Update hidden-to-output weights.
  for (i = 0; i < this.hiddenCount + 1; i++) {
    for (j = 0; j < this.outputCount; j++) {
      delta = eta * outputGradients[j] * (i === 0 ? 1.0 : this.ihOutputs[i - 1]);
      this.hoWeights[i][j] += delta + alpha * this.hoPrevWeightsDelta[i][j];
      this.hoPrevWeightsDelta[i][j] = delta;
    }
  }
};

NeuralNetwork.prototype.setWeights = function (weights) {
  var totalWeights = this.totalWeightCount();
  if (weights.length !== totalWeights) {
    throw new Error('Different length!');
  }
  var offset = 0;

  this.ihWeights = matrix.reshape(weights, this.inputCount + 1, this.hiddenCount, offset);
  offset += (this.inputCount + 1) * this.hiddenCount;

  this.hoWeights = matrix.reshape(weights, this.hiddenCount + 1, this.outputCount, offset);
};

NeuralNetwork.prototype.getWeights = function () {
  var ih = matrix.flatten(this.ihWeights);
  var ho = matrix.flatten(this.hoWeights);
  return [].concat(ih, ho);
};

NeuralNetwork.prototype.totalWeightCount = function () {
  return (this.inputCount + 1) * this.hiddenCount +
    (this.hiddenCount + 1) * this.outputCount;
};

module.exports = NeuralNetwork;
