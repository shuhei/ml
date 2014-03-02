'use strict';

var expect = require('chai').expect;
var vector = require('../vector');

describe('vector', function () {
  describe('product', function () {
    it ('calculate the product of matrices', function () {
      var vec = [1.2, 2, 3];
      var another = [4, 5, 6];
      var prod = vector.product(vec, another);
      expect(prod).to.deep.equal(32.8);
    });
  });
});
