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

function size(mat) {
  if (mat.length === 0 || mat[0].length === 0) {
    throw new Error('Invalid matrix!');
  }
  var rows = mat.length;
  var cols = mat[0].length;
  return [rows, cols];
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

function zeros(rows, cols) {
  return make(rows, cols, 0);
}

function ones(rows, cols) {
  return make(rows, cols, 1);
}

// Concat matrices horizontally.
function hConcat(left, right) {
  if (left.length !== right.length) {
    throw new Error('Different rows!');
  }
  var rows = left.length;
  var leftCols = left[0].length;
  var rightCols = right[0].length;
  var result = make(rows, leftCols + rightCols);
  var i, j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < leftCols; j++) {
      result[i][j] = left[i][j];
    }
    for (j = 0; j < rightCols; j++) {
      result[i][leftCols + j] = right[i][j];
    }
  }
  return result;
}

function vConcat(top, bottom) {
  if (top[0].length !== bottom[0].length) {
    throw new Error('Different cols!');
  }
  return [].concat(top, bottom);
}

function map(mat, func) {
  var s = size(mat);
  var rows = s[0];
  var cols = s[1];
  var result = make(rows, cols);
  var i, j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      result[i][j] = func(mat[i][j]);
    }
  }
  return result;
}

function transpose(mat) {
  var s = size(mat);
  var rows = s[0];
  var cols = s[1];
  var result = make(cols, rows);
  var i, j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      result[j][i] = mat[i][j];
    }
  }
  return result;
}

function applyMatrix(left, right) {
  var leftSize = size(left);
  var rightSize = size(right);
  if (leftSize[1] !== rightSize[0]) {
    throw new Error('Imcompatible size! left: ' + leftSize + ' right: ' + rightSize);
  }
  var rows = leftSize[0];
  var cols = rightSize[1];
  var result = make(rows, cols);
  var i, j, k, sum;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      sum = 0;
      for (k = 0; k < leftSize[1]; k++) {
        sum += left[i][k] * right[k][j];
      }
      result[i][j] = sum;
    }
  }
  return result;
}

function pickColumn(mat, col) {
  var result = vector.make(mat.length);
  var i;
  for (i = 0; i < mat.length; i++) {
    result[i] = mat[i][col];
  }
  return result;
}

function pickColumns(mat, colStart, numCols) {
  numCols = numCols || mat[0].length - colStart;
  var result = make(mat.length, numCols);
  var i, j;
  for (i = 0; i < mat.length; i++) {
    for (j = 0; j < numCols; j++) {
      result[i][j] = mat[i][colStart + j];
    }
  }
  return result;
}

module.exports = {
  make: make,
  size: size,
  flatten: flatten,
  reshape: reshape,
  zeros: zeros,
  ones: ones,
  hConcat: hConcat,
  vConcat: vConcat,
  transpose: transpose,
  applyMatrix: applyMatrix,
  map: map,
  pickColumn: pickColumn,
  pickColumns: pickColumns
};
