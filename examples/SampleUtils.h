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

#ifndef WEBNN_NATIVE_EXAMPLES_SAMPLE_UTILS_H_
#define WEBNN_NATIVE_EXAMPLES_SAMPLE_UTILS_H_

#include <dawn/webnn.h>
#include <dawn/webnn_cpp.h>
#include <condition_variable>
#include <mutex>
#include <vector>

#include "common/RefCounted.h"

uint32_t product(const std::vector<int32_t>& dims);

wnn::NeuralNetworkContext CreateCppNeuralNetworkContext();

wnn::NamedInputs CreateCppNamedInputs();

wnn::NamedOperands CreateCppNamedOperands();

wnn::NamedOutputs CreateCppNamedOutputs();

bool Expected(float output, float expected);

namespace utils {

    class WrappedModel : public RefCounted {
      public:
        WrappedModel() = default;
        ~WrappedModel() = default;

        void SetInput(std::vector<int32_t> shape, std::vector<float> buffer);
        wnn::OperandDescriptor* InputDesc();
        std::vector<float> InputBuffer();

        void SetConstant(std::vector<int32_t> shape, std::vector<float> buffer);
        wnn::OperandDescriptor* ConstantDesc();
        void const* ConstantBuffer();
        size_t ConstantLength();

        virtual wnn::Operand GenerateOutput(wnn::ModelBuilder nn);
        void SetOutputShape(std::vector<int32_t> shape);
        std::vector<int32_t> OutputShape();

        void SetExpectedBuffer(std::vector<float> buffer);
        std::vector<float> ExpectedBuffer();

        void SetExpectedShape(std::vector<int32_t> shape);
        std::vector<int32_t> ExpectedShape();

        void SetOutputExpected(bool expected);

        bool GetOutputExpected();

      private:
        wnn::ModelBuilder model_builder_;
        std::vector<int32_t> input_shape_;
        std::vector<float> input_buffer_;
        wnn::OperandDescriptor input_desc_;
        wnn::OperandDescriptor constant_desc_;
        std::vector<int32_t> constant_shape_;
        std::vector<float> constant_buffer_;
        std::vector<int32_t> output_shape_;
        std::vector<float> expected_buffer_;
        std::vector<int32_t> expected_shape_;
        bool output_expected_ = true;
    };

    bool Test(WrappedModel* model);

    class ComputeSync {
      public:
        ComputeSync() : done_(false) {
        }
        ~ComputeSync() = default;
        void Wait();
        void Finish();

      private:
        std::condition_variable cond_var_;
        std::mutex mutex_;
        bool done_;
    };

}  // namespace utils

#endif  // WEBNN_NATIVE_EXAMPLES_SAMPLE_UTILS_H_
