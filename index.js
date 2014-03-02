'use strict';

// Coursera Machine Learning ex4

var fs = require('fs');
var matrix = require('./matrix');
var nnCostFunction = require('./nn-cost-function');

function parseCSV(csv) {
  var lines = csv.split("\n").filter(Boolean);
  return lines.map(function(line) {
    return line.split(',').map(function (str) {
      return parseFloat(str, 10);
    });
  });
}

function loadCSV(filename) {
  return parseCSV(fs.readFileSync(filename, 'utf8'));
}

var X = loadCSV('training.csv');
var y = loadCSV('answer.csv').map(function (row) { return row[0] });
console.log('Training data:', matrix.size(X));

var inputLayerSize = 400; // 20x20 input image.
var hiddenLayerSize = 25; // 25 units. Not 25 layers.
var numLabels = 10;

// Already trained Theta.
var Theta1 = loadCSV('theta1.csv');
var Theta2 = loadCSV('theta2.csv');

console.log('Theta1:', Theta1.length, Theta1[0].length);
console.log('Theta2:', Theta2.length, Theta2[0].length);

var nnParams = [].concat(matrix.flatten(Theta1), matrix.flatten(Theta2));
console.log('NN params:', nnParams.length);

// Weight regularization parameter.
var lambda = 0;

var result = nnCostFunction(nnParams, inputLayerSize, hiddenLayerSize,
  numLabels, X, y, lambda);
var J = result[0];
var gradients = result[1];

console.log('Cost (should be around 0.383770):', J);
