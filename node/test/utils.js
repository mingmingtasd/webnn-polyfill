'use strict';
const assert = require('assert').strict;

function almostEqual(
    a, b, episilon = 1e-6, rtol = 5.0 * 1.1920928955078125e-7) {
  const delta = Math.abs(a - b);
  if (delta <= episilon + rtol * Math.abs(b)) {
    return true;
  } else {
    console.warn(`a(${a}) b(${b}) delta(${delta})`);
    return false;
  }
}

function checkValue(output, expected) {
  assert.ok(output.length === expected.length);
  for (let i = 0; i < output.length; ++i) {
    assert.ok(almostEqual(output[i], expected[i]));
  }
}

function sizeOfShape(array) {
  return array.reduce(
      (accumulator, currentValue) => accumulator * currentValue);
}

function checkShape(shape, expected) {
  assert.isTrue(shape.length === expected.length);
  for (let i = 0; i < shape.length; ++i) {
    assert.isTrue(shape[i] === expected[i]);
  }
}

module.exports = {
  checkValue
}