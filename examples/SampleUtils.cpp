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

wnn::NeuralNetworkContext CreateCppNeuralNetworkContext() {
  DawnProcTable backendProcs = dawn_native::GetProcs();
  dawnProcSetProcs(&backendProcs);
  return wnn::NeuralNetworkContext::Acquire(dawn_native::CreateNeuralNetworkContext());
}

wnn::NamedInputs CreateCppNamedInputs() {
  return wnn::NamedInputs::Acquire(dawn_native::CreateNamedInputs());
}

wnn::NamedOperands CreateCppNamedOperands() {
  return wnn::NamedOperands::Acquire(dawn_native::CreateNamedOperands());
}

wnn::NamedOutputs CreateCppNamedOutputs() {
  return wnn::NamedOutputs::Acquire(dawn_native::CreateNamedOutputs());
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

void WrappedModel::SetExpectedShape(std::vector<int32_t> shape) {
  expected_shape_ = std::move(shape);
}

std::vector<int32_t> WrappedModel::ExpectedShape() { return expected_shape_; }

void WrappedModel::SetExpectedBuffer(std::vector<float> buffer) {
  expected_buffer_ = std::move(buffer);
}

std::vector<float> WrappedModel::ExpectedBuffer() { return expected_buffer_; }

wnn::Operand WrappedModel::GenerateOutput(wnn::ModelBuilder nn) {
  UNREACHABLE();
}

wnn::NeuralNetworkContext g_context;
WrappedModel *g_wrapped_model;
void ComputeCallback(WNNComputeStatus status, WNNNamedResults impl,
    char const * message, void* userData) {
  if (status != WNNComputeStatus_Success) {
    dawn::ErrorLog() << message;
    return;
  }
  wnn::NamedResults outputs = outputs.Acquire(impl);
  wnn::Result output = outputs.Get("output");
  std::vector<float> expected_data = g_wrapped_model->ExpectedBuffer();
  bool expected = true;
  for (size_t i = 0; i < output.BufferSize() / sizeof(float); ++i) {
    float output_data = static_cast<const float *>(output.Buffer())[i];
    if (!Expected(output_data, expected_data[i])) {
      dawn::ErrorLog() << "The output doesn't output as expected for "
                       << output_data << " != " << expected_data[i]
                       << " index = " << i;
      expected = false;
      break;
    }
  }
  std::vector<int32_t> expected_shape = g_wrapped_model->ExpectedShape();
  if (!expected_shape.empty()) {
    for (size_t i = 0; i < output.DimensionsSize(); ++i) {
      int32_t dimension = output.Dimensions()[i];
      if (!Expected(expected_shape[i], dimension)) {
        dawn::ErrorLog() << "The output dimension is not as expected for "
                       << dimension << " != " << expected_shape[i]
                       << " index = " << i;
        expected = false;
        break;
      }
    }
  }
  if (expected) {
    dawn::InfoLog() << "The output output as expected.";
  }
  delete g_wrapped_model;
  g_context = nullptr;
}

void CompilationCallback(WNNCompileStatus status, WNNCompilation impl,
    char const * message, void* userData) {
  if (status != WNNCompileStatus_Success) {
    dawn::ErrorLog() << message;
    return;
  }
  wnn::Compilation exe = exe.Acquire(impl);

  std::vector<float> input_buffer = g_wrapped_model->InputBuffer();
  wnn::Input a;
  a.buffer = input_buffer.data();
  a.size = input_buffer.size() * sizeof(float);
  wnn::NamedInputs inputs = CreateCppNamedInputs();
  inputs.Set("input", &a);

  exe.Compute(inputs, ComputeCallback, nullptr, nullptr);
}

void ErrorCallback(WNNErrorType type, char const * message, void * userdata) {
  if (type != WNNErrorType_NoError) {
    dawn::ErrorLog() << "error type is " << type << ", messages are " << message;
  }
}

// Wrapped Compilation
void Test(WrappedModel *wrapped_model) {
  g_wrapped_model = wrapped_model;
  g_context = CreateCppNeuralNetworkContext();
  g_context.SetUncapturedErrorCallback(ErrorCallback, nullptr);

  wnn::ModelBuilder builder = g_context.CreateModelBuilder();
  wnn::Operand output_operand = wrapped_model->GenerateOutput(builder);
  wnn::NamedOperands named_operands = CreateCppNamedOperands();
  named_operands.Set("output", output_operand);
  // Use Promise in JS to await callback. 
  g_context.PushErrorScope(wnn::ErrorFilter::Validation);
  wnn::Model model = builder.CreateModel(named_operands);
  g_context.PopErrorScope(ErrorCallback, nullptr);
  model.Compile(CompilationCallback, nullptr);
}

} // namespace utils
