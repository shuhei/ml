'use strict';

function make(length, initialValue) {
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

function copy(from, to) {
  if (from.length !== to.length) {
    throw new Error('Different size!');
  }
  for (var i = 0; i < from.length; i++) {
    to[i] = from[i];
  }
}

function add(from, to) {
  if (from.length !== to.length) {
    throw new Error('Different size!');
  }
  for (var i = 0; i < from.length; i++) {
    to[i] += from[i];
  }
}

function applyMatrix(vector, matrix) {
  var len = matrix[0].length;
  var result = make(len);
  var i, j;
  for (i = 0; i < vector.length; i++) {
    for (j = 0; j < len; j++) {
      result[j] += vector[i] * matrix[i][j];
    }
  }
  return result;
}

function product(left, right) {
  if (left.length !== right.length) {
    throw new Error('Different size!');
  }
  var sum = 0;
  var i;
  for (i = 0; i < left.length; i++) {
    sum += left[i] * right[i];
  }
  return sum;
}

module.exports = {
  make: make,
  copy: copy,
  add: add,
  product: product,
  applyMatrix: applyMatrix
};
