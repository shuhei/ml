'use strict';

function tanh(x) {
  return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
}

// Used as the input-to-hidden activation function.
function sigmoid(x) {
  if (x < -45.0) return 0.0;
  else if (x > 45.0) return 1.0;
  else return 1.0 / (1.0 + Math.exp(-x));
}

function hyperTanh(x) {
  if (x < -10.0) return -1.0;
  else if (x > 10.0) return 1.0;
  else return tanh(x);
}

module.exports = {
  sigmoid: sigmoid,
  hyperTanh: hyperTanh
};
