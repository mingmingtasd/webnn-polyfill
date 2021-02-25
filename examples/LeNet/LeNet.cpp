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

#include "examples/LeNet/LeNet.h"

#include <stdlib.h>
#include <chrono>

#include "common/Log.h"
#include "examples/SampleUtils.h"

const size_t WEIGHTS_LENGTH = 1724336;

LeNet::LeNet() : mWeightsData(new char[WEIGHTS_LENGTH]), mValidationFailed(false) {
    mContext = CreateCppNeuralNetworkContext();
    mContext.SetUncapturedErrorCallback(LeNet::UncapturedErrorCallback, this);
}

bool LeNet::Load(const std::string& weigthsPath) {
    FILE* fp = fopen(weigthsPath.c_str(), "rb");
    if (!fp) {
        dawn::ErrorLog() << "Failed to open weights file at " << weigthsPath << ".";
        return false;
    }

    size_t size = fread(mWeightsData.get(), sizeof(char), WEIGHTS_LENGTH, fp);
    fclose(fp);
    if (size != WEIGHTS_LENGTH) {
        dawn::ErrorLog() << "The expected size of weights file is " << WEIGHTS_LENGTH
                         << ", but got " << size;
        return false;
    }

    webnn::ModelBuilder builder = mContext.CreateModelBuilder();

    mContext.PushErrorScope(webnn::ErrorFilter::Validation);
    uint32_t byteOffset = 0;
    std::vector<int32_t> inputShape = {1, 1, 28, 28};
    webnn::OperandDescriptor inputDesc = {webnn::OperandType::Float32, inputShape.data(),
                                          (uint32_t)inputShape.size()};

    webnn::Operand input = builder.Input("input", &inputDesc);

    std::vector<int32_t> conv2d1FilterShape = {20, 1, 5, 5};
    float* conv2d1FilterData = reinterpret_cast<float*>(mWeightsData.get() + byteOffset);
    byteOffset += product(conv2d1FilterShape) * sizeof(float);
    webnn::OperandDescriptor conv2d1FilterDesc = {webnn::OperandType::Float32,
                                                  conv2d1FilterShape.data(),
                                                  (uint32_t)conv2d1FilterShape.size()};
    webnn::Operand conv2d1FilterConstant = builder.Constant(
        &conv2d1FilterDesc, conv2d1FilterData, product(conv2d1FilterShape) * sizeof(float));
    webnn::Conv2dOptions conv2d1Options = {};
    webnn::Operand conv1 = builder.Conv2d(input, conv2d1FilterConstant, &conv2d1Options);

    std::vector<int32_t> add1BiasShape = {1, 20, 1, 1};
    float* add1BiasData = reinterpret_cast<float*>(mWeightsData.get() + byteOffset);
    byteOffset += product(add1BiasShape) * sizeof(float);
    webnn::OperandDescriptor add1BiasDesc = {webnn::OperandType::Float32, add1BiasShape.data(),
                                             (uint32_t)add1BiasShape.size()};
    webnn::Operand add1BiasConstant =
        builder.Constant(&add1BiasDesc, add1BiasData, product(add1BiasShape) * sizeof(float));
    webnn::Operand add1 = builder.Add(conv1, add1BiasConstant);

    webnn::Pool2dOptions options = {};
    std::vector<int32_t> windowDimensions = {2, 2};
    options.windowDimensions = windowDimensions.data();
    std::vector<int32_t> strides = {2, 2};
    options.strides = strides.data();
    options.stridesCount = 2;
    std::vector<int32_t> padding = {0, 0, 0, 0};
    options.padding = padding.data();
    options.paddingCount = 4;
    webnn::Operand pool1 = builder.MaxPool2d(add1, &options);

    std::vector<int32_t> conv2d2FilterShape = {50, 20, 5, 5};
    float* conv2d2FilterData = reinterpret_cast<float*>(mWeightsData.get() + byteOffset);
    byteOffset += product(conv2d2FilterShape) * sizeof(float);
    webnn::OperandDescriptor conv2d2FilterDesc = {webnn::OperandType::Float32,
                                                  conv2d2FilterShape.data(),
                                                  (uint32_t)conv2d2FilterShape.size()};
    webnn::Operand conv2d2FilterConstant = builder.Constant(
        &conv2d2FilterDesc, conv2d2FilterData, product(conv2d2FilterShape) * sizeof(float));
    webnn::Conv2dOptions conv2d2Options = {};
    webnn::Operand conv2 = builder.Conv2d(pool1, conv2d2FilterConstant, &conv2d2Options);

    std::vector<int32_t> add2BiasShape = {1, 50, 1, 1};
    float* add2BiasData = reinterpret_cast<float*>(mWeightsData.get() + byteOffset);
    byteOffset += product(add2BiasShape) * sizeof(float);
    webnn::OperandDescriptor add2BiasDesc = {webnn::OperandType::Float32, add2BiasShape.data(),
                                             (uint32_t)add2BiasShape.size()};
    webnn::Operand add2BiasConstant =
        builder.Constant(&add2BiasDesc, add2BiasData, product(add2BiasShape) * sizeof(float));
    webnn::Operand add2 = builder.Add(conv2, add2BiasConstant);

    webnn::Pool2dOptions options2 = {};
    std::vector<int32_t> windowDimensions2 = {2, 2};
    options2.windowDimensions = windowDimensions2.data();
    std::vector<int32_t> strides2 = {2, 2};
    options2.strides = strides2.data();
    options2.stridesCount = 2;
    std::vector<int32_t> padding2 = {0, 0, 0, 0};
    options2.padding = padding2.data();
    options2.paddingCount = 4;
    webnn::Operand pool2 = builder.MaxPool2d(add2, &options2);

    std::vector<int32_t> newShape = {1, -1};
    webnn::Operand reshape1 = builder.Reshape(pool2, newShape.data(), newShape.size());
    // skip the new shape, 2 int64 values
    byteOffset += 2 * 8;

    std::vector<int32_t> matmulShape = {500, 800};
    float* matmulData = reinterpret_cast<float*>(mWeightsData.get() + byteOffset);
    byteOffset += product(matmulShape) * sizeof(float);
    webnn::OperandDescriptor matmulDataDesc = {webnn::OperandType::Float32, matmulShape.data(),
                                               (uint32_t)matmulShape.size()};
    webnn::Operand matmulWeights =
        builder.Constant(&matmulDataDesc, matmulData, product(matmulShape) * sizeof(float));

    webnn::Operand matmul1WeightsTransposed = builder.Transpose(matmulWeights);
    webnn::Operand matmul1 = builder.Matmul(reshape1, matmul1WeightsTransposed);

    std::vector<int32_t> add3BiasShape = {1, 500};
    float* add3BiasData = reinterpret_cast<float*>(mWeightsData.get() + byteOffset);
    byteOffset += product(add3BiasShape) * sizeof(float);
    webnn::OperandDescriptor add3BiasDesc = {webnn::OperandType::Float32, add3BiasShape.data(),
                                             (uint32_t)add3BiasShape.size()};
    webnn::Operand add3BiasConstant =
        builder.Constant(&add3BiasDesc, add3BiasData, product(add3BiasShape) * sizeof(float));
    webnn::Operand add3 = builder.Add(matmul1, add3BiasConstant);

    webnn::Operand relu = builder.Relu(add3);

    std::vector<int32_t> newShape2 = {1, -1};
    webnn::Operand reshape2 = builder.Reshape(relu, newShape2.data(), newShape2.size());

    std::vector<int32_t> matmulShape2 = {10, 500};
    float* matmulData2 = reinterpret_cast<float*>(mWeightsData.get() + byteOffset);
    byteOffset += product(matmulShape2) * sizeof(float);
    webnn::OperandDescriptor matmulData2Desc = {webnn::OperandType::Float32, matmulShape2.data(),
                                                (uint32_t)matmulShape2.size()};
    webnn::Operand matmulWeights2 =
        builder.Constant(&matmulData2Desc, matmulData2, product(matmulShape2) * sizeof(float));

    webnn::Operand matmul1WeightsTransposed2 = builder.Transpose(matmulWeights2);
    webnn::Operand matmul2 = builder.Matmul(reshape2, matmul1WeightsTransposed2);

    std::vector<int32_t> add4BiasShape = {1, 10};
    float* add4BiasData = reinterpret_cast<float*>(mWeightsData.get() + byteOffset);
    byteOffset += product(add4BiasShape) * sizeof(float);
    webnn::OperandDescriptor add4BiasDesc = {webnn::OperandType::Float32, add4BiasShape.data(),
                                             (uint32_t)add4BiasShape.size()};
    webnn::Operand add4BiasConstant =
        builder.Constant(&add4BiasDesc, add4BiasData, product(add4BiasShape) * sizeof(float));
    webnn::Operand add4 = builder.Add(matmul2, add4BiasConstant);

    webnn::Operand softmax = builder.Softmax(add4);

    webnn::NamedOperands namedOperands = CreateCppNamedOperands();
    namedOperands.Set("output", softmax);
    mModel = builder.CreateModel(namedOperands);
    mContext.PopErrorScope(LeNet::ValidationErrorCallback, this);
    // FIXME: wait for LeNet::ValidationErrorCallback sets mValidationFailed.
    if (mValidationFailed) {
        mModel = webnn::Model();
        return false;
    }
    return true;
}

bool LeNet::Compile(webnn::CompilationOptions const* options) {
    if (!mModel) {
        dawn::ErrorLog() << "Model is not ready.";
        return false;
    }
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime =
        std::chrono::high_resolution_clock::now();
    mModel.Compile(LeNet::CompilationCallback, this, options);
    // FIXME: wait for LeNet::CompilationCallback sets the mCompilation.
    if (!mCompilation) {
        return false;
    }
    std::chrono::duration<double, std::milli> elapsedTime =
        std::chrono::high_resolution_clock::now() - startTime;
    dawn::InfoLog() << "Compile Time: " << elapsedTime.count() << " ms";
    return true;
}

webnn::Result LeNet::Compute(const void* inputData, size_t inputLength) {
    if (!mCompilation) {
        dawn::ErrorLog() << "Compilation is not ready.";
        return webnn::Result();
    }
    webnn::Input input;
    input.buffer = inputData;
    input.size = inputLength;
    webnn::NamedInputs inputs = CreateCppNamedInputs();
    inputs.Set("input", &input);
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime =
        std::chrono::high_resolution_clock::now();
    mCompilation.Compute(inputs, LeNet::ComputeCallback, this, nullptr);
    // FIXME: wait for LeNet::ComputeCallback sets the mOutputs.
    if (!mOutputs) {
        return webnn::Result();
    }
    std::chrono::duration<double, std::milli> elapsedTime =
        std::chrono::high_resolution_clock::now() - startTime;
    dawn::InfoLog() << "Compute time: " << elapsedTime.count() << " ms";
    return mOutputs.Get("output");
}

void LeNet::UncapturedErrorCallback(WebnnErrorType type, char const* message, void* userData) {
    dawn::ErrorLog() << "UncapturedError type is " << type << ", message is " << message;
    DAWN_ASSERT(type == WebnnErrorType_NoError);
}

void LeNet::ValidationErrorCallback(WebnnErrorType type, char const* message, void* userData) {
    LeNet* lenet = reinterpret_cast<LeNet*>(userData);
    DAWN_ASSERT(lenet);
    if (type != WebnnErrorType_NoError) {
        dawn::ErrorLog() << "ValidationError type is " << type << ", message is " << message;
        lenet->mValidationFailed = true;
    }
}

void LeNet::CompilationCallback(WebnnCompileStatus status,
                                WebnnCompilation impl,
                                char const* message,
                                void* userData) {
    LeNet* lenet = reinterpret_cast<LeNet*>(userData);
    DAWN_ASSERT(lenet);
    if (status != WebnnCompileStatus_Success) {
        dawn::ErrorLog() << "Compile failed: " << message;
        return;
    }
    lenet->mCompilation = lenet->mCompilation.Acquire(impl);
    return;
}

void LeNet::ComputeCallback(WebnnComputeStatus status,
                            WebnnNamedResults impl,
                            char const* message,
                            void* userData) {
    LeNet* lenet = reinterpret_cast<LeNet*>(userData);
    DAWN_ASSERT(lenet);
    if (status != WebnnComputeStatus_Success) {
        dawn::ErrorLog() << "Compute failed: " << message;
        return;
    }
    lenet->mOutputs = lenet->mOutputs.Acquire(impl);
    return;
}
