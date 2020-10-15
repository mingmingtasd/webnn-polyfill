// Copyright 2017 The Dawn Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "SampleUtils.h"

#include <stdio.h>
#include <vector>
#include "common/Log.h"

bool Expected(float output, float expected) {
  return (fabs(output - expected) < 0.005f);
}

void compute_callback(WNNOutputs impl) { 
  wnn::Outputs outputs = outputs.Acquire(impl);
  wnn::Output output = outputs.GetOutput("output");
  std::vector<float> expected_data = {1.5347629 , -0.3981255 ,  2.6510081 , -0.14295794,  0.6647107 , -0.70315295,  1.3096018 ,  3.9256358 ,  3.873897};
  for (size_t i = 0; i < output.size; ++i) {
    float output_data = static_cast<float*>(output.buffer)[i];
    if (!Expected(output_data, expected_data[i])) {
      dawn::ErrorLog() << "The output doesn't output as expected for " << output_data << " != " << expected_data[i];
      return;
    }
  }
  dawn::InfoLog() << "The output output as expected. ";
}

void compilation_callback(WNNCompilation impl) {
  wnn::Compilation exe = exe.Acquire(impl);
  std::vector<float> bufferA = {0.9602246 ,  0.97682184, -0.33201018,  0.8248904 ,  0.40872088,
        0.18995902,  0.69355214, -0.37210146,  0.18104352,  3.270753  ,
        -0.803097  , -0.7268995};
  wnn::Input a;
  a.buffer = bufferA.data();
  a.size = bufferA.size();
  wnn::Inputs inputs = CreateCppInputs();
  inputs.SetInput("a", &a);
  wnn::Outputs outputs = CreateCppOutputs();
  std::vector<float> outputA(3*3, 1.0);
  wnn::Output output;
  output.buffer = outputA.data();
  output.size = outputA.size();
  outputs.SetOutput("output", &output);
  exe.Compute(inputs, compute_callback, outputs);
}

int main(int argc, const char* argv[]) {
  wnn::ModelBuilder nn = CreateCppModelBuilder();
  std::vector<int32_t> shapeA = {3, 4};
  wnn::OperandDescriptor descA = {wnn::OperandType::Float32, shapeA.data(),
                                  (uint32_t)shapeA.size()};
  wnn::Operand a = nn.Input("a", &descA);
  std::vector<int32_t> shapeB = {4, 3};
  wnn::OperandDescriptor descB = {wnn::OperandType::Float32, shapeB.data(),
                                  (uint32_t)shapeB.size()};
  std::vector<float> bufferB = {0.17467105, -1.2045133 , -0.02621938,  0.6096196 ,  1.4499376 ,
            1.3465316 ,  0.03289436,  1.0754977 , -0.61485314,  0.94857556,
            -0.36462623,  1.402278};
  wnn::Operand b = nn.Constant(&descB, bufferB.data(), 4*3);
  wnn::Operand c = nn.Matmul(a, b);
  wnn::NamedOperand operand = {"output", c};
  wnn::Model model = nn.CreateModel(&operand, 1);
  model.Compile(compilation_callback);
}
