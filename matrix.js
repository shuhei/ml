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

module.exports = {
  make: make
};
