'use strict';

var expect = require('chai').expect;
var math = require('../math');

describe('math', function () {
  describe('sigmoid', function () {
    it ('calculates sigmoid', function () {
      expect(math.sigmoid(1)).to.equal(0.7310585786300049);
    });
  });
});
