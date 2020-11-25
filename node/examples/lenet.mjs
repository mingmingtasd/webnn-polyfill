'use strict';

import WebNN from "../index.js";
import utils from './utils.js';

Object.assign(global, WebNN);

(async function main() {
  const context = ML.getNeuralNetworkContext();

  const builder = context.createModelBuilder();
  const a = builder.input('a', {type: 'float32', dimensions: [3, 4]});
  const b = builder.constant(
      {type: 'float32', dimensions: [4, 3]}, new Float32Array([0.17467105, -1.2045133, -0.02621938, 0.6096196, 1.4499376, 1.3465316, 0.03289436, 1.0754977, -0.61485314, 0.94857556, -0.36462623, 1.402278]));
  const c = builder.add(a, b);
  const model = builder.createModel({c});
  const compiledModel = await model.compile();
  const inputs = {
    'a': {
      buffer: new Float32Array([0.9602246, 0.97682184, -0.33201018, 0.8248904, 0.40872088, 0.18995902, 0.69355214, -0.37210146, 0.18104352, 3.270753, -0.803097, -0.7268995]),
    },
  };
  const outputs = await compiledModel.compute(inputs);
  const expected = [1.5347629, -0.3981255, 2.6510081, -0.14295794, 0.6647107, -0.70315295, 1.3096018, 3.9256358, 3.873897];
  utils.checkValue(outputs.buffer, expected);
  console.log("sucess for outputs buffer " + outputs.buffer);
})();
