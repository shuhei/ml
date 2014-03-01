'use strict';

var vector = require('./vector');

function make(rows, cols, initialValue) {
  var result = new Array(rows);
  var i;
  if (initialValue === undefined) {
    initialValue = 0;
  }
  for (i = 0; i < rows; i++) {
    result[i] = vector.make(cols, initialValue);
  }
  return result;
}

function flatten(mat) {
  var rows = mat.length;
  var cols = mat[0].length;
  var result = vector.make(rows * cols);
  var i, j;
  var k = 0;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      result[k++] = mat[i][j];
    }
  }
  return result;
}

function reshape(vec, rows, cols, offset) {
  var result = make(rows, cols);
  var i, j;
  var k = offset || 0;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      result[i][j] = vec[k++];
    }
  }
  return result;
}

module.exports = {
  make: make,
  flatten: flatten,
  reshape: reshape
};
