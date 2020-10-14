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

void compute_callback(WNNOutputs impl) {
  printf("outputs %p\n", (void*)impl);
}

void compilation_callback(WNNCompilation impl) {
  wnn::Compilation exe;
  exe.Acquire(impl);
  std::vector<float> bufferA(2*3);
  wnn::Input a;
  a.buffer = bufferA.data();
  a.size = bufferA.size();
  wnn::Inputs inputs;
  inputs.SetInput("a", &a);
  wnn::Outputs outputs;
  exe.Compute(inputs, compute_callback, outputs);
}

int main(int argc, const char* argv[]) {
  wnn::NeuralNetworkContext nn = CreateCppNeuralNetworkContext();
  std::vector<int32_t> shapeA = {2, 3};
  wnn::OperandDescriptor descA = {wnn::OperandType::Float32, shapeA.data(), (uint32_t)shapeA.size()};
  wnn::Operand a = nn.Input("a", &descA);
  std::vector<int32_t> shapeB = {3, 2};
  wnn::OperandDescriptor descB = {wnn::OperandType::Float32, shapeB.data(), (uint32_t)shapeB.size()};
  std::vector<float> bufferB(3*2);
  wnn::Operand b = nn.Constant(&descB, bufferB.data(), bufferB.size(), 0);
  wnn::Operand c = nn.Matmul(a, b);
  wnn::Model model = nn.CreateModel();
  model.Compile(compilation_callback);
}
