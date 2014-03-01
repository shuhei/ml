'use strict';

var NeuralNetwork = require('./neural-network');

function calculateError(target, output) {
  var sum = 0.0;
  for (var i = 0; i < target.length; i++) {
    sum += Math.abs(target[i] - output[i]);
  }
  return sum;
}

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
