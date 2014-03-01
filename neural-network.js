var vector = require('./vector');
var matrix = require('./matrix');

// http://msdn.microsoft.com/en-us/magazine/jj658979.aspx

function tanh(x) {
  return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
}

// Used as the input-to-hidden activation function.
function sigmoid(x) {
  if (x < -45.0) return 0.0;
  else if (x > 45.0) return 1.0;
  else return 1.0 / (1.0 + Math.exp(-x));
}

function hyperTan(x) {
  if (x < -10.0) return -1.0;
  else if (x > 10.0) return 1.0;
  else return tanh(x);
}

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
  this.ihOutputs = ihSums.map(sigmoid);

  // TODO: Why different functions?
  var hoSums = vector.applyMatrix([1.0].concat(this.ihOutputs), this.hoWeights);
  this.outputs = hoSums.map(hyperTan);

  var result = vector.make(this.outputCount);
  vector.copy(this.outputs, result);
  return result;
};

NeuralNetwork.prototype.updateWeights = function (tValues, eta, alpha) {
  var i;
  var derivative, sum, delta;
  var outputGradients = vector.make(this.outputCount);
  var hiddenGradients = vector.make(this.hiddenCount);

  if (tValues.length !== this.outputCount) {
    throw new Error('Target values not same length as output.');
  }

  // Compute output gradients
  for (i = 0; i < this.outputCount; i++) {
    derivative = (1 - this.outputs[i]) * (1 + this.outputs[i]);
    outputGradients[i] = derivative * (tValues[i] - this.outputs[i]);
  }

  // Compute hidden gradients
  for (i = 0; i < this.hiddenCount; i++) {
    derivative = (1 - this.ihOutputs[i]) * this.ihOutputs[i];
    sum = 0.0;
    for (j = 0; j < this.outputCount; j++) {
      sum += outputGradients[j] * this.hoWeights[i + 1][j];
    }
    hiddenGradients[i] = derivative * sum;
  }

  // Update input-to-hidden weights
  for (i = 0; i < this.inputCount + 1; i++) {
    for (j = 0; j < this.hiddenCount; j++) {
      // TODO: This is confusing. Add bias to inputs?
      delta = eta * hiddenGradients[j] * (i === 0 ? 1.0 : this.inputs[i - 1]);
      this.ihWeights[i][j] += delta + alpha * this.ihPrevWeightsDelta[i][j];
      this.ihPrevWeightsDelta[i][j] = delta;
    }
  }

  // Update hidden-to-output weights
  for (i = 0; i < this.hiddenCount + 1; i++) {
    for (j = 0; j < this.outputCount; j++) {
      // TODO: This is confusing. Add bias to ihOutputs?
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

  var i, j;
  var k = 0;

  for (i = 0; i < this.inputCount + 1; i++) {
    for (j = 0; j < this.hiddenCount; j++) {
      this.ihWeights[i][j] = weights[k++];
    }
  }

  for (i = 0; i < this.hiddenCount + 1; i++) {
    for (j = 0; j < this.outputCount; j++) {
      this.hoWeights[i][j] = weights[k++];
    }
  }
};

NeuralNetwork.prototype.getWeights = function () {
  var totalWeights = this.totalWeightCount();
  var result = vector.make(totalWeights);
  var k = 0;
  var i, j;

  // TODO: Extract copyArray with offset and flatten matrix.
  for (i = 0; i < this.inputCount + 1; i++) {
    for (j = 0; j < this.hiddenCount; j++) {
      result[k++] = this.ihWeights[i][j];
    }
  }

  for (i = 0; i < this.hiddenCount + 1; i++) {
    for (j = 0; j < this.outputCount; j++) {
      result[k++] = this.hoWeights[i][j];
    }
  }

  return result;
};

NeuralNetwork.prototype.totalWeightCount = function () {
  return (this.inputCount + 1) * this.hiddenCount +
    (this.hiddenCount + 1) * this.outputCount;
};

function calculateError(target, output) {
  var sum = 0.0;
  for (var i = 0; i < target.length; i++) {
    sum += Math.abs(target[i] - output[i]);
  }
  return sum;
}

function main() {
  var nn = new NeuralNetwork(3, 4, 2);

  // Arbitrary weights.
  var weights = [
    // Input to hidden
    -2.0, -6.0, -1.0, -7.0,
    0.1, 0.2, 0.3, 0.4,
    0.5, 0.6, 0.7, 0.8,
    0.9, 1.0, 1.1, 1.2,

    // Hidden to output
    -2.5, -5.0,
    1.3, 1.4,
    1.5, 1.6,
    1.7, 1.8,
    1.9, 2.0
  ];
  nn.setWeights(weights);

  // Training data.
  var xValues = [1.0, 2.0, 3.0]; // Input values
  var tValues = [-0.8500, 0.7500]; // Target values

  var eta = 0.90; // Learning rate
  var alpha = 0.04;

  // Start training.
  var counter = 0;
  var yValues = nn.computeOutputs(xValues);
  var error = calculateError(tValues, yValues);
  console.log('Initial error:', error);

  while (counter < 1000 && error > 0.01) {
    nn.updateWeights(tValues, eta, alpha);
    yValues = nn.computeOutputs(xValues);
    error = calculateError(tValues, yValues);
    counter++;
  }

  // Training done.
  console.log('Counter:', counter);
  console.log('Minimized error:', error);
  var bestWeights = nn.getWeights();
  console.log(bestWeights);
}

main();
