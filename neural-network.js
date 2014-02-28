// http://msdn.microsoft.com/en-us/magazine/jj658979.aspx

function makeArray(length, initialValue) {
  var result = new Array(length);
  var i;
  if (initialValue === undefined) {
    initialValue = 0;
  }
  for (i = 0; i < length; i++) {
    result[i] = initialValue;
  }
  return result;
}

function makeMatrix(rows, cols, initialValue) {
  var result = new Array(rows);
  var i;
  if (initialValue === undefined) {
    initialValue = 0;
  }
  for (i = 0; i < rows; i++) {
    result[i] = makeArray(cols, initialValue);
  }
  return result;
}

function copyArray(from, to) {
  if (from.length !== to.length) {
    throw new Error('Different length!');
  }
  for (var i = 0; i < from.length; i++) {
    to[i] = from[i];
  }
}

function addArray(from, to) {
  if (from.length !== to.length) {
    throw new Error('Different length!');
  }
  for (var i = 0; i < from.length; i++) {
    to[i] += from[i];
  }
}

function NeuralNetwork(inputCount, hiddenCount, outputCount) {
  this.inputCount = inputCount;
  this.hiddenCount = hiddenCount;
  this.outputCount = outputCount;

  this.inputs = makeArray(inputCount);

  this.ihWeights = makeMatrix(inputCount, hiddenCount);
  this.ihSums = makeArray(hiddenCount);
  this.ihBiases = makeArray(hiddenCount);

  this.ihOutputs = makeArray(hiddenCount);

  this.hoWeights = makeMatrix(hiddenCount, outputCount);
  this.hoSums = makeArray(outputCount);
  this.hoBiases = makeArray(outputCount);

  this.outputs = makeArray(outputCount);

  this.oGrads = makeArray(outputCount);
  this.hGrads = makeArray(hiddenCount);

  this.ihPrevWeightsDelta = makeMatrix(inputCount, hiddenCount);
  this.ihPrevBiasesDelta = makeArray(hiddenCount);
  this.hoPrevWeightsDelta = makeMatrix(hiddenCount, outputCount);
  this.hoPrevBiasesDelta = makeArray(outputCount);
}

NeuralNetwork.prototype.updateWeights = function (tValues, eta, alpha) {
  var i;
  var derivative, sum, delta;

  if (tValues.length !== this.outputCount) {
    throw new Error('Target values not same length as output.');
  }

  // Compute output gradients
  for (i = 0; i < this.outputCount; i++) {
    derivative = (1 - this.outputs[i]) * (1 + this.outputs[i]);
    this.oGrads[i] = derivative * (tValues[i] - this.outputs[i]);
  }

  // Compute hidden gradients
  for (i = 0; i < this.ihWeights.length; i++) {
    derivative = (1 - this.ihOutputs[i]) * this.ihOutputs[i];
    sum = 0.0;
    for (j = 0; j < this.outputCount; j++) {
      sum += this.oGrads[j] * this.hoWeights[i][j];
    }
    this.hGrads[i] = derivative * sum;
  }

  // Update input-to-hidden weights
  for (i = 0; i < this.inputCount; i++) {
    for (j = 0; j < this.hiddenCount; j++) {
      delta = eta * this.hGrads[j] * this.inputs[i];
      this.ihWeights[i][j] += delta;
      this.ihWeights[i][j] += alpha * this.ihPrevWeightsDelta[i][j];
    }
  }

  // Update input-to-hidden biases
  for (i = 0; i < this.hiddenCount; i++) {
    delta = eta * this.hGrads[i] * 1.0;
    this.ihBiases[i] += delta;
    this.ihBiases[i] += alpha * this.ihPrevBiasesDelta[i];
  }

  // Update hidden-to-output weights
  for (i = 0; i < this.hiddenCount; i++) {
    for (j = 0; j < this.outputCoun; j++) {
      delta = eta * this.oGrads[j] * this.ihOutputs[i];
      this.hoWeights[i][j] += delta;
      this.hoWeights[i][j] += alpha * this.hoPrevWeightsDelta[i][j];
      this.hoPrevWeightsDelta[i][j] = delta;
    }
  }

  // Update hidden-to-output biases
  for (i = 0; i < this.outputCount; i++) {
    delta = eta * this.oGrads[i] * 1.0;
    this.hoBiases[i] += delta;
    this.hoBiases[i] += alpha * this.hoPrevBiasesDelta[i];
    this.hoPrevBiasesDelta[i] = delta;
  }
};

NeuralNetwork.prototype.setWeights = function (weights) {
  var totalWeights = this.totalWeightCount();
  if (weights.length !== totalWeights) {
    throw new Error('Different length!');
  }

  var i, j;
  var k = 0;

  for (i = 0; i < this.inputCount; i++) {
    for (j = 0; j < this.hiddenCount; j++) {
      this.ihWeights[i][j] = weights[k++];
    }
  }

  for (i = 0; i < this.hiddenCount; i++) {
    this.ihBiases[i] = weights[k++];
  }

  for (i = 0; i < this.hiddenCount; i++) {
    for (j = 0; j < this.outputCount; j++) {
      this.hoWeights[i][j] = weights[k++];
    }
  }

  for (i = 0; i < this.outputCount; i++) {
    this.hoBiases[i] = weights[k++];
  }
};

NeuralNetwork.prototype.getWeights = function () {
  var totalWeights = this.totalWeightCount();
  var result = makeArray(totalWeights);
  var k = 0;
  var i, j;

  // TODO: Extract copyArray with offset and flatten matrix.
  for (i = 0; i < this.inputCount; i++) {
    for (j = 0; j < this.hiddenCount; j++) {
      result[k++] = this.ihWeights[i][j];
    }
  }

  for (i = 0; i < this.hiddenCount; i++) {
    result[k++] = this.ihBiases[i];
  }

  for (i = 0; i < this.hiddenCount; i++) {
    for (j = 0; j < this.outputCount; j++) {
      result[k++] = this.hoWeights[i][j];
    }
  }

  for (i = 0; i < this.outputCount; i++) {
    result[k++] = this.hoBiases[i];
  }

  return result;
};

NeuralNetwork.prototype.totalWeightCount = function () {
  return this.inputCount * this.hiddenCount +
    this.hiddenCount * this.outputCount +
    this.hiddenCount + this.outputCount;
};

NeuralNetwork.prototype.computeOutputs = function (xValues) {
  var i, j;
  if (xValues.length !== this.inputCount) {
    throw new Error('Expected ' + this.inputCount + ' inputs but got ' + xValues.length);
  }

  // TODO: Extract fill zero function.
  for (i = 0; i < this.hiddenCount; i++) {
    this.ihSums[i] = 0.0;
  }
  for (i = 0; i < this.outputCount; i++) {
    this.hoSums[i] = 0.0;
  }

  copyArray(xValues, this.inputs);

  // TODO: Extract matrix functios.
  // Compute input-to-hidden weighted sums.
  for (j = 0; j < this.hiddenCount; j++) {
    for (i = 0; i < this.inputCount; i++) {
      this.ihSums[j] += this.inputs[i] * this.ihWeights[i][j];
    }
  }

  // Add biases to input-to-hidden sums.
  addArray(this.ihBiases, this.ihSums);

  // Determine input-to-hidden output.
  for (i = 0; i < this.hiddenCount; i++) {
    this.ihOutputs[i] = this.sigmoidFunction(this.ihSums[i]);
  }

  // TODO: Extract the pattern of the three.
  // TODO: Why different functions?

  for (j = 0; j < this.outputCount; j++) {
    for (i = 0; i < this.hiddenCount; i++) {
      this.hoSums[j] += this.ihOutputs[i] * this.hoWeights[i][j];
    }
  }

  addArray(this.hoBiases, this.hoSums);

  for (i = 0; i < this.outputCount; i++) {
    this.outputs[i] = this.hyperTanFunction(this.hoSums[i]);
  }

  var result = makeArray(this.outputCount);
  copyArray(this.outputs, result);
  return result;
};

// Used as the input-to-hidden activation function.
NeuralNetwork.prototype.sigmoidFunction = function (x) {
  if (x < -45.0) return 0.0;
  else if (x > 45.0) return 1.0;
  else return 1.0 / (1.0 + Math.exp(-x));
};

NeuralNetwork.prototype.hyperTanFunction = function (x) {
  if (x < -10.0) return -1.0;
  else if (x > 10.0) return 1.0;
  else return tanh(x);
};

function tanh(x) {
  return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
}

function calculateError(target, output) {
  var sum = 0.0;
  for (var i = 0; i < target.length; i++) {
    sum += Math.abs(target[i] - output[i]);
  }
  return sum;
}

function main() {
  var nn = new NeuralNetwork(3, 4, 2);
  // Arbitrary weights and biases.
  var weights = [
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
    -2.0, -6.0, -1.0, -7.0,
    1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
    -2.5, -5.0
  ];
  nn.setWeights(weights);
  var xValues = [1.0, 2.0, 3.0];
  var initialOutputs = nn.computeOutputs(xValues);
  var tValues = [-0.8500, 0.7500]; // Target values
  var eta = 0.90; // Learning rate
  var alpha = 0.04;
  var counter = 0;
  var yValues = nn.computeOutputs(xValues);
  var error = calculateError(tValues, yValues);
  while (counter < 1000 && error > 0.01) {
    nn.updateWeights(tValues, eta, alpha);
    yValues = nn.computeOutputs(xValues);
    error = calculateError(tValues, yValues);
    console.log(error);
    counter++;
  }
  var bestWeights = nn.getWeights();
  console.log(bestWeights);
}

main();
