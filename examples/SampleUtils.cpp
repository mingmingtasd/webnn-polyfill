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

#include <webnn/webnn.h>
#include <webnn/webnn_cpp.h>
#include <webnn/webnn_proc.h>
#include <webnn_native/WebnnNative.h>
#include "common/Assert.h"
#include "common/Log.h"

uint32_t product(const std::vector<int32_t>& dims) {
    uint32_t prod = 1;
    for (size_t i = 0; i < dims.size(); ++i)
        prod *= dims[i];
    return prod;
}

webnn::NeuralNetworkContext CreateCppNeuralNetworkContext() {
    WebnnProcTable backendProcs = webnn_native::GetProcs();
    webnnProcSetProcs(&backendProcs);
    WebnnNeuralNetworkContext context = webnn_native::CreateNeuralNetworkContext();
    if (context) {
        return webnn::NeuralNetworkContext::Acquire(context);
    }
    return webnn::NeuralNetworkContext();
}

webnn::NamedInputs CreateCppNamedInputs() {
    return webnn::NamedInputs::Acquire(webnn_native::CreateNamedInputs());
}

webnn::NamedOperands CreateCppNamedOperands() {
    return webnn::NamedOperands::Acquire(webnn_native::CreateNamedOperands());
}

webnn::NamedOutputs CreateCppNamedOutputs() {
    return webnn::NamedOutputs::Acquire(webnn_native::CreateNamedOutputs());
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
        mInputDesc = {webnn::OperandType::Float32, mInputShape.data(),
                      (uint32_t)mInputShape.size()};
    }

    webnn::OperandDescriptor* WrappedModel::InputDesc() {
        return &mInputDesc;
    }

    std::vector<float> WrappedModel::InputBuffer() {
        return mInputBuffer;
    }

    void WrappedModel::SetConstant(std::vector<int32_t> shape, std::vector<float> buffer) {
        mConstantShape = std::move(shape);
        mConstantBuffer = std::move(buffer);
        mConstantDesc = {webnn::OperandType::Float32, mConstantShape.data(),
                         (uint32_t)mConstantShape.size()};
    }

    webnn::OperandDescriptor* WrappedModel::ConstantDesc() {
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

    webnn::Operand WrappedModel::GenerateOutput(webnn::ModelBuilder nn) {
        UNREACHABLE();
    }

    void WrappedModel::SetComputedResult(bool expected) {
        mOutputExpected = expected;
    }

    bool WrappedModel::GetComputedResult() {
        return mOutputExpected;
    }

    // The Compilation should be released unitl ComputeCallback.
    webnn::Compilation gCompilation;
    WrappedModel* gWrappedModel;

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

    void ComputeCallback(WebnnComputeStatus status,
                         WebnnNamedResults impl,
                         char const* message,
                         void* userData) {
        if (status != WebnnComputeStatus_Success) {
            dawn::InfoLog() << "Test failed.";
            dawn::ErrorLog() << message;
            return;
        }
        webnn::NamedResults outputs = outputs.Acquire(impl);
        webnn::Result output = outputs.Get("output");
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
        return;
    }

    void CompilationCallback(WebnnCompileStatus status,
                             WebnnCompilation impl,
                             char const* message,
                             void* userData) {
        if (status != WebnnCompileStatus_Success) {
            dawn::ErrorLog() << message;
            return;
        }

        std::vector<float> inputBuffer = gWrappedModel->InputBuffer();
        webnn::Input a;
        a.buffer = inputBuffer.data();
        a.size = inputBuffer.size() * sizeof(float);
        webnn::NamedInputs inputs = CreateCppNamedInputs();
        inputs.Set("input", &a);
        gCompilation = gCompilation.Acquire(impl);
        gCompilation.Compute(inputs, ComputeCallback, nullptr, nullptr);
    }

    void ErrorCallback(WebnnErrorType type, char const* message, void* userdata) {
        if (type != WebnnErrorType_NoError) {
            dawn::ErrorLog() << "error type is " << type << ", messages are " << message;
        }
    }

    // Wrapped Compilation
    bool Test(WrappedModel* wrappedModel) {
        gWrappedModel = wrappedModel;
        // TODO(mingming): Move the code of creating context to setup with reusing ValidationTest
        // class so that end_to_end test body will not run if fail to create context.
        webnn::NeuralNetworkContext context = CreateCppNeuralNetworkContext();
        context.SetUncapturedErrorCallback(ErrorCallback, nullptr);

        webnn::ModelBuilder builder = context.CreateModelBuilder();
        webnn::Operand outputOperand = wrappedModel->GenerateOutput(builder);
        webnn::NamedOperands namedOperands = CreateCppNamedOperands();
        namedOperands.Set("output", outputOperand);
        // Use Promise in JS to await callback.
        context.PushErrorScope(webnn::ErrorFilter::Validation);
        webnn::Model model = builder.CreateModel(namedOperands);
        context.PopErrorScope(ErrorCallback, nullptr);
        model.Compile(CompilationCallback, nullptr);

        bool expected = wrappedModel->GetComputedResult();
        // Release backend resources in main thread.
        delete gWrappedModel;
        gCompilation = nullptr;
        return expected;
    }

}  // namespace utils
