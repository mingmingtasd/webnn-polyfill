'use strict';
const webNN = require("../../index");
const utils = require("../utils");

describe('test reshape', function() {
  const nn = webNN.ML.getNeuralNetworkContext();

  async function testReshape(oldShape, newShape, expectedShape) {
    const builder = nn.createModelBuilder();
    const x = builder.input('x', {type: 'float32', dimensions: oldShape});
    const y = builder.reshape(x, newShape);
    const model = builder.createModel({y});
    const compiledModel = await model.compile();
    const bufferSize = utils.sizeOfShape(oldShape);
    const inputBuffer = new Float32Array(bufferSize);
    for (let i = 0; i < inputBuffer.length; ++i) {
      inputBuffer[i] = Math.random();
    }
    const inputs = {'x': {buffer: inputBuffer}};
    const outputs = await compiledModel.compute(inputs);
    // utils.checkShape(
    //     outputs.y.dimensions, expectedShape ? expectedShape : newShape);
    utils.checkValue(outputs.buffer, inputBuffer);
  }

  it('reshape reordered_all_dims', async function() {
    testReshape([2, 3, 4], [4, 2, 3]);
  });

  it('reshape reordered_last_dims', async function() {
    testReshape([2, 3, 4], [2, 4, 3]);
  });

  it('reshape reduced_dims', async function() {
    testReshape([2, 3, 4], [2, 12]);
  });

  it('reshape extended_dims', async function() {
    testReshape([2, 3, 4], [2, 3, 2, 2]);
  });

  it('reshape one_dim', async function() {
    testReshape([2, 3, 4], [24]);
  });

  it('reshape negative_dim', async function() {
    testReshape([2, 3, 4], [2, -1, 2], [2, 6, 2]);
  });

  it('reshape negative_dim', async function() {
    testReshape([2, 3, 4], [-1, 2, 3, 4], [1, 2, 3, 4]);
  });
});
