'use strict';

import WebNN from "../index.js";
import utils from './utils.js';

Object.assign(global, WebNN);

(async function main() {
  const context = ML.getNeuralNetworkContext();

  const builder = context.createModelBuilder();
  const input = builder.input('a', {type: 'float32', dimensions: [1, 1, 5, 5]});
  const filter = builder.constant(
      {type: 'float32', dimensions: [1, 1, 3, 3]}, new Float32Array(9).fill(1));
  const output = builder.conv2d(input, filter, {padding});
  const model = builder.createModel({output});
  const compiledModel = await model.compile();
  const inputs = {
    'input': {
      buffer: new Float32Array([0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,]),
    },
  };
  const outputs = await compiledModel.compute(inputs);
  const expected = [ 12.,  21., 27., 33.,  24.,  33.,  54.,  63., 72.,  51.,  63.,  99., 108.,
    117., 81., 93., 144., 153., 162., 111., 72., 111., 117., 123., 84.,];
  utils.checkValue(outputs.buffer, expected);
  console.log("sucess for outputs buffer " + outputs.buffer);
})();
