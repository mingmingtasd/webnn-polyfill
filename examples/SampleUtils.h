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
        WrappedModel();
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

        void SetComputedResult(bool expected);

        bool GetComputedResult();

      private:
        wnn::ModelBuilder mModelBuilder;
        std::vector<int32_t> mInputShape;
        std::vector<float> mInputBuffer;
        wnn::OperandDescriptor mInputDesc;
        wnn::OperandDescriptor mConstantDesc;
        std::vector<int32_t> mConstantShape;
        std::vector<float> mConstantBuffer;
        std::vector<int32_t> mOutputShape;
        std::vector<float> mExpectedBuffer;
        std::vector<int32_t> mExpectedShape;
        bool mOutputExpected;
    };

    bool Test(WrappedModel* model);

    class ComputeSync {
      public:
        ComputeSync() : mDone(false) {
        }
        ~ComputeSync() = default;
        void Wait();
        void Finish();

      private:
        std::condition_variable mCondVar;
        std::mutex mMutex;
        bool mDone;
    };

}  // namespace utils

#endif  // WEBNN_NATIVE_EXAMPLES_SAMPLE_UTILS_H_
