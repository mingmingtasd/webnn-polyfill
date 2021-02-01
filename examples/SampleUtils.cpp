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

#include <webnn/webnn_proc.h>
#include <webnn/webnn.h>
#include <webnn/webnn_cpp.h>
#include <webnn_native/WebnnNative.h>
#include "common/Assert.h"
#include "common/Log.h"

uint32_t product(const std::vector<int32_t>& dims) {
    uint32_t prod = 1;
    for (size_t i = 0; i < dims.size(); ++i)
        prod *= dims[i];
    return prod;
}

wnn::NeuralNetworkContext CreateCppNeuralNetworkContext() {
    WebnnProcTable backendProcs = webnn_native::GetProcs();
    webnnProcSetProcs(&backendProcs);
    return wnn::NeuralNetworkContext::Acquire(webnn_native::CreateNeuralNetworkContext());
}

wnn::NamedInputs CreateCppNamedInputs() {
    return wnn::NamedInputs::Acquire(webnn_native::CreateNamedInputs());
}

wnn::NamedOperands CreateCppNamedOperands() {
    return wnn::NamedOperands::Acquire(webnn_native::CreateNamedOperands());
}

wnn::NamedOutputs CreateCppNamedOutputs() {
    return wnn::NamedOutputs::Acquire(webnn_native::CreateNamedOutputs());
}

bool Expected(float output, float expected) {
    return (fabs(output - expected) < 0.005f);
}

namespace utils {

    WrappedModel::WrappedModel() : mOutputExpected(true) {
    }

    void WrappedModel::SetInput(std::vector<int32_t> shape, std::vector<float> buffer) {
        mInputShape = std::move(shape);
        mInputBuffer = std::move(buffer);
        mInputDesc = {wnn::OperandType::Float32, mInputShape.data(), (uint32_t)mInputShape.size()};
    }

    wnn::OperandDescriptor* WrappedModel::InputDesc() {
        return &mInputDesc;
    }

    std::vector<float> WrappedModel::InputBuffer() {
        return mInputBuffer;
    }

    void WrappedModel::SetConstant(std::vector<int32_t> shape, std::vector<float> buffer) {
        mConstantShape = std::move(shape);
        mConstantBuffer = std::move(buffer);
        mConstantDesc = {wnn::OperandType::Float32, mConstantShape.data(),
                         (uint32_t)mConstantShape.size()};
    }

    wnn::OperandDescriptor* WrappedModel::ConstantDesc() {
        return &mConstantDesc;
    }

    void const* WrappedModel::ConstantBuffer() {
        return mConstantBuffer.data();
    }

    size_t WrappedModel::ConstantLength() {
        return mConstantBuffer.size() * sizeof(float);
    }

    void WrappedModel::SetOutputShape(std::vector<int32_t> shape) {
        mOutputShape = std::move(shape);
    }

    std::vector<int32_t> WrappedModel::OutputShape() {
        return mOutputShape;
    }

    void WrappedModel::SetExpectedShape(std::vector<int32_t> shape) {
        mExpectedShape = std::move(shape);
    }

    std::vector<int32_t> WrappedModel::ExpectedShape() {
        return mExpectedShape;
    }

    void WrappedModel::SetExpectedBuffer(std::vector<float> buffer) {
        mExpectedBuffer = std::move(buffer);
    }

    std::vector<float> WrappedModel::ExpectedBuffer() {
        return mExpectedBuffer;
    }

    wnn::Operand WrappedModel::GenerateOutput(wnn::ModelBuilder nn) {
        UNREACHABLE();
    }

    void WrappedModel::SetComputedResult(bool expected) {
        mOutputExpected = expected;
    }

    bool WrappedModel::GetComputedResult() {
        return mOutputExpected;
    }

    // The Compilation should be released unitl ComputeCallback.
    wnn::Compilation gCompilation;
    WrappedModel* gWrappedModel;
    ComputeSync gComputeSync;

    void ComputeSync::Wait() {
        // Wait for async callback.
        std::unique_lock<std::mutex> lock(mMutex);
        bool& done = mDone;
        mCondVar.wait(lock, [&done] { return done; });
        mDone = false;
    }

    void ComputeSync::Finish() {
        std::lock_guard<std::mutex> lock(mMutex);
        mDone = true;
        mCondVar.notify_one();
        return;
    }

    void ComputeCallback(WNNComputeStatus status,
                         WNNNamedResults impl,
                         char const* message,
                         void* userData) {
        if (status != WNNComputeStatus_Success) {
            dawn::InfoLog() << "Test failed.";
            dawn::ErrorLog() << message;
            gComputeSync.Finish();
            return;
        }
        wnn::NamedResults outputs = outputs.Acquire(impl);
        wnn::Result output = outputs.Get("output");
        std::vector<float> expectedData = gWrappedModel->ExpectedBuffer();
        bool expected = true;
        for (size_t i = 0; i < output.BufferSize() / sizeof(float); ++i) {
            float outputData = static_cast<const float*>(output.Buffer())[i];
            if (!Expected(outputData, expectedData[i])) {
                dawn::ErrorLog() << "The output doesn't output as expected for " << outputData
                                 << " != " << expectedData[i] << " index = " << i;
                expected = false;
                break;
            }
        }
        std::vector<int32_t> expectedShape = gWrappedModel->ExpectedShape();
        if (!expectedShape.empty()) {
            if (expectedShape.size() != output.DimensionsSize()) {
                expected = false;
                dawn::ErrorLog() << "The output rank is not as expected for "
                                 << expectedShape.size() << " != " << output.DimensionsSize();
            } else {
                for (size_t i = 0; i < output.DimensionsSize(); ++i) {
                    int32_t dimension = output.Dimensions()[i];
                    if (!Expected(expectedShape[i], dimension)) {
                        dawn::ErrorLog()
                            << "The output dimension is not as expected for " << dimension
                            << " != " << expectedShape[i] << " index = " << i;
                        expected = false;
                        break;
                    }
                }
            }
        }
        gWrappedModel->SetComputedResult(expected);
        gComputeSync.Finish();
        return;
    }

    void CompilationCallback(WNNCompileStatus status,
                             WNNCompilation impl,
                             char const* message,
                             void* userData) {
        if (status != WNNCompileStatus_Success) {
            dawn::ErrorLog() << message;
            gComputeSync.Finish();
            return;
        }

        std::vector<float> inputBuffer = gWrappedModel->InputBuffer();
        wnn::Input a;
        a.buffer = inputBuffer.data();
        a.size = inputBuffer.size() * sizeof(float);
        wnn::NamedInputs inputs = CreateCppNamedInputs();
        inputs.Set("input", &a);
        gCompilation = gCompilation.Acquire(impl);
        gCompilation.Compute(inputs, ComputeCallback, nullptr, nullptr);
    }

    void ErrorCallback(WNNErrorType type, char const* message, void* userdata) {
        if (type != WNNErrorType_NoError) {
            dawn::ErrorLog() << "error type is " << type << ", messages are " << message;
        }
    }

    // Wrapped Compilation
    bool Test(WrappedModel* wrappedModel) {
        gWrappedModel = wrappedModel;
        wnn::NeuralNetworkContext context = CreateCppNeuralNetworkContext();
        context.SetUncapturedErrorCallback(ErrorCallback, nullptr);

        wnn::ModelBuilder builder = context.CreateModelBuilder();
        wnn::Operand outputOperand = wrappedModel->GenerateOutput(builder);
        wnn::NamedOperands namedOperands = CreateCppNamedOperands();
        namedOperands.Set("output", outputOperand);
        // Use Promise in JS to await callback.
        context.PushErrorScope(wnn::ErrorFilter::Validation);
        wnn::Model model = builder.CreateModel(namedOperands);
        context.PopErrorScope(ErrorCallback, nullptr);
        model.Compile(CompilationCallback, nullptr);

        gComputeSync.Wait();
        bool expected = wrappedModel->GetComputedResult();
        // Release backend resources in main thread.
        delete gWrappedModel;
        gCompilation = nullptr;
        return expected;
    }

}  // namespace utils
