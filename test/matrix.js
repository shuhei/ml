'use strict';

var expect = require('chai').expect;
var matrix = require('../matrix');

describe('matrix', function () {
  describe('reshape', function () {
    it ('creates a matrix from the vector', function () {
      var mat = matrix.reshape([1, 2, 3, 4, 5, 6], 2, 3);
      expect(mat).to.deep.equal([[1, 2, 3], [4, 5, 6]]);
    });
  });

  describe('transpose', function () {
    it ('transposes the matrix', function () {
      var transposed = matrix.transpose([[1, 2, 3], [4, 5, 6]]);
      expect(transposed).to.deep.equal([[1, 4], [2, 5], [3, 6]]);
    });
  });

  describe('applyMatrix', function () {
    it ('applies another matrix to the matrix', function () {
      var mat = [[1, 2, 3], [4, 5, 6]];
      var another = [[1, 2], [3, 4], [5, 6]];
      var applied = matrix.applyMatrix(mat, another);
      expect(applied).to.deep.equal([[22, 28], [49, 64]]);
    });
  });

  describe('pickColumn', function () {
    it('creates a column vector', function () {
      var mat = [[1, 2, 3], [4, 5, 6]];
      var picked = matrix.pickColumn(mat, 1);
      expect(picked).to.deep.equal([2, 5]);
    });
  });

  describe('pickColumns', function () {
    it('picks a sub matrix', function () {
      var mat = [[1, 2, 3], [4, 5, 6]];
      var picked = matrix.pickColumns(mat, 1);
      expect(picked).to.deep.equal([[2, 3], [5, 6]]);
    });
  });
});
