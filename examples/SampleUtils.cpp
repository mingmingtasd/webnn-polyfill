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

#include "common/Assert.h"
#include "common/Log.h"
#include <dawn/dawn_proc.h>
#include <dawn/webnn.h>
#include <dawn/webnn_cpp.h>
#include <dawn_native/DawnNative.h>

uint32_t product(const std::vector<int32_t> &dims) {
  uint32_t prod = 1;
  for (size_t i = 0; i < dims.size(); ++i)
    prod *= dims[i];
  return prod;
}

wnn::ModelBuilder CreateCppModelBuilder() {
  DawnProcTable backendProcs = dawn_native::GetProcs();
  dawnProcSetProcs(&backendProcs);
  dawn_native::NeuralNetworkContext context;
  return wnn::ModelBuilder::Acquire(context.CreateModelBuilder());
}

wnn::Inputs CreateCppInputs() {
  return wnn::Inputs::Acquire(dawn_native::CreateInputs());
}

wnn::Outputs CreateCppOutputs() {
  return wnn::Outputs::Acquire(dawn_native::CreateOutputs());
}

bool Expected(float output, float expected) {
  return (fabs(output - expected) < 0.005f);
}

namespace utils {

void WrappedModel::SetInput(std::vector<int32_t> shape,
                            std::vector<float> buffer) {
  input_shape_ = std::move(shape);
  input_buffer_ = std::move(buffer);
  input_desc_ = {wnn::OperandType::Float32, input_shape_.data(),
                 (uint32_t)input_shape_.size()};
}

wnn::OperandDescriptor *WrappedModel::InputDesc() { return &input_desc_; }

std::vector<float> WrappedModel::InputBuffer() { return input_buffer_; }

void WrappedModel::SetConstant(std::vector<int32_t> shape,
                               std::vector<float> buffer) {
  constant_shape_ = std::move(shape);
  constant_buffer_ = std::move(buffer);
  constant_desc_ = {wnn::OperandType::Float32, constant_shape_.data(),
                    (uint32_t)constant_shape_.size()};
}

wnn::OperandDescriptor *WrappedModel::ConstantDesc() { return &constant_desc_; }

void const *WrappedModel::ConstantBuffer() { return constant_buffer_.data(); }

size_t WrappedModel::ConstantLength() { return constant_buffer_.size() * sizeof(float); }

void WrappedModel::SetOutputShape(std::vector<int32_t> shape) {
  output_shape_ = std::move(shape);
}

std::vector<int32_t> WrappedModel::OutputShape() { return output_shape_; }

void WrappedModel::SetExpectedBuffer(std::vector<float> buffer) {
  expected_buffer_ = std::move(buffer);
}

std::vector<float> WrappedModel::ExpectedBuffer() { return expected_buffer_; }

wnn::Operand WrappedModel::GenerateOutput(wnn::ModelBuilder nn) {
  UNREACHABLE();
}

WrappedModel *s_wrapped_model;
void ComputeCallback(WNNOutputs impl, void* userData) {}

void CompilationCallback(WNNCompilation impl, void* userData) {
  wnn::Compilation exe = exe.Acquire(impl);

  std::vector<float> input_buffer = s_wrapped_model->InputBuffer();
  wnn::Input a;
  a.buffer = input_buffer.data();
  a.size = input_buffer.size() * sizeof(float);
  wnn::Inputs inputs = CreateCppInputs();
  inputs.SetInput("input", &a);

  wnn::Outputs outputs = exe.Compute(inputs, ComputeCallback, nullptr, nullptr);
  wnn::Output output = outputs.GetOutputWithIndex(0);
  std::vector<float> expected_data = s_wrapped_model->ExpectedBuffer();
  bool expected = true;
  for (size_t i = 0; i < output.size / sizeof(float); ++i) {
    float output_data = static_cast<float *>(output.buffer)[i];
    if (!Expected(output_data, expected_data[i])) {
      dawn::ErrorLog() << "The output doesn't output as expected for "
                       << output_data << " != " << expected_data[i]
                       << " index = " << i;
      expected = false;
      break;
    }
  }
  if (expected) {
    dawn::InfoLog() << "The output output as expected.";
  }
  delete s_wrapped_model;
}

// Wrapped Compilation
void Test(WrappedModel *wrapped_model) {
  s_wrapped_model = wrapped_model;
  wnn::ModelBuilder nn = CreateCppModelBuilder();
  wnn::Operand output_operand = wrapped_model->GenerateOutput(nn);
  wnn::NamedOperand named_operand = {"output", output_operand};
  wnn::Model model = nn.CreateModel(&named_operand, 1);
  model.Compile(CompilationCallback, nullptr);
}

} // namespace utils
