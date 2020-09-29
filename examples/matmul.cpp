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

void compute_callback(WGPUOutputs impl) {
  printf("outputs %p\n", (void*)impl);
}

void compilation_callback(WGPUCompilation impl) {
  wgpu::Compilation exe;
  exe.Acquire(impl);
  std::vector<float> bufferA(2*3);
  wgpu::Input a;
  a.buffer = bufferA.data();
  a.size = bufferA.size();
  wgpu::Inputs inputs;
  inputs.SetInput("a", &a);
  wgpu::Outputs outputs;
  exe.Compute(inputs, compute_callback, outputs);
}

int main(int argc, const char* argv[]) {
  wgpu::NeuralNetworkContext nn = CreateCppNeuralNetworkContext();
  std::vector<int32_t> shapeA = {2, 3};
  wgpu::OperandDescriptor descA = {wgpu::OperandType::Float32, shapeA.data(), (uint32_t)shapeA.size()};
  wgpu::Operand a = nn.Input("a", &descA);
  std::vector<int32_t> shapeB = {3, 2};
  wgpu::OperandDescriptor descB = {wgpu::OperandType::Float32, shapeB.data(), (uint32_t)shapeB.size()};
  std::vector<float> bufferB(3*2);
  wgpu::Operand b = nn.Constant(&descB, bufferB.data(), bufferB.size(), 0);
  wgpu::Operand c = nn.Matmul(a, b);
  wgpu::Model model = nn.CreateModel();
  model.Compile(compilation_callback);
}
