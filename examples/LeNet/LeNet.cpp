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

const size_t WEIGHTS_LENGTH = 1724336;

LeNet::LeNet() {
    mContext = CreateCppNeuralNetworkContext();
    mContext.SetUncapturedErrorCallback(
        [](WebnnErrorType type, char const* message, void* userData) {
            if (type != WebnnErrorType_NoError) {
                dawn::ErrorLog() << "Error type is " << type << ", message is " << message;
            }
        },
        this);
}

bool LeNet::Load(const std::string& weigthsPath) {
    FILE* fp = fopen(weigthsPath.c_str(), "rb");
    if (!fp) {
        dawn::ErrorLog() << "Failed to open weights file at " << weigthsPath << ".";
        return false;
    }

    std::unique_ptr<char> weightsData(new char[WEIGHTS_LENGTH]);
    const size_t readSize = fread(weightsData.get(), sizeof(char), WEIGHTS_LENGTH, fp);
    fclose(fp);
    if (readSize != WEIGHTS_LENGTH) {
        dawn::ErrorLog() << "The expected size of weights file is " << WEIGHTS_LENGTH
                         << ", but got " << readSize;
        return false;
    }

    const webnn::ModelBuilder builder = mContext.CreateModelBuilder();

    uint32_t byteOffset = 0;
    const webnn::Operand input = utils::BuildInput(builder, "input", {1, 1, 28, 28});

    const std::vector<int32_t> conv2d1FilterShape = {20, 1, 5, 5};
    const float* conv2d1FilterData = reinterpret_cast<float*>(weightsData.get() + byteOffset);
    const uint32_t conv2d1FilterDataLength = product(conv2d1FilterShape) * sizeof(float);
    byteOffset += conv2d1FilterDataLength;
    const webnn::Operand conv2d1FilterConstant = utils::BuildConstant(
        builder, conv2d1FilterShape, conv2d1FilterData, conv2d1FilterDataLength);
    const webnn::Operand conv1 = builder.Conv2d(input, conv2d1FilterConstant);

    const std::vector<int32_t> add1BiasShape = {1, 20, 1, 1};
    const float* add1BiasData = reinterpret_cast<float*>(weightsData.get() + byteOffset);
    const uint32_t add1BiasDataLength = product(add1BiasShape) * sizeof(float);
    byteOffset += add1BiasDataLength;
    const webnn::Operand add1BiasConstant =
        utils::BuildConstant(builder, add1BiasShape, add1BiasData, add1BiasDataLength);
    const webnn::Operand add1 = builder.Add(conv1, add1BiasConstant);

    utils::Pool2dOptions pool1Options;
    pool1Options.windowDimensions = {2, 2};
    pool1Options.strides = {2, 2};
    const webnn::Operand pool1 = builder.MaxPool2d(add1, pool1Options.AsPtr());

    const std::vector<int32_t> conv2d2FilterShape = {50, 20, 5, 5};
    const float* conv2d2FilterData = reinterpret_cast<float*>(weightsData.get() + byteOffset);
    const uint32_t conv2d2FilterDataLength = product(conv2d2FilterShape) * sizeof(float);
    byteOffset += conv2d2FilterDataLength;
    const webnn::Operand conv2d2FilterConstant = utils::BuildConstant(
        builder, conv2d2FilterShape, conv2d2FilterData, conv2d2FilterDataLength);
    const webnn::Operand conv2 = builder.Conv2d(pool1, conv2d2FilterConstant);

    const std::vector<int32_t> add2BiasShape = {1, 50, 1, 1};
    const float* add2BiasData = reinterpret_cast<float*>(weightsData.get() + byteOffset);
    const uint32_t add2BiasDataLength = product(add2BiasShape) * sizeof(float);
    byteOffset += add2BiasDataLength;
    const webnn::Operand add2BiasConstant =
        utils::BuildConstant(builder, add2BiasShape, add2BiasData, add2BiasDataLength);
    const webnn::Operand add2 = builder.Add(conv2, add2BiasConstant);

    utils::Pool2dOptions pool2Options;
    pool2Options.windowDimensions = {2, 2};
    pool2Options.strides = {2, 2};
    const webnn::Operand pool2 = builder.MaxPool2d(add2, pool2Options.AsPtr());

    const std::vector<int32_t> newShape = {1, -1};
    const webnn::Operand reshape1 = builder.Reshape(pool2, newShape.data(), newShape.size());
    // skip the new shape, 2 int64 values
    byteOffset += 2 * 8;

    const std::vector<int32_t> matmul1Shape = {500, 800};
    const float* matmul1Data = reinterpret_cast<float*>(weightsData.get() + byteOffset);
    const uint32_t matmul1DataLength = product(matmul1Shape) * sizeof(float);
    byteOffset += matmul1DataLength;
    const webnn::Operand matmul1Weights =
        utils::BuildConstant(builder, matmul1Shape, matmul1Data, matmul1DataLength);
    const webnn::Operand matmul1WeightsTransposed = builder.Transpose(matmul1Weights);
    const webnn::Operand matmul1 = builder.Matmul(reshape1, matmul1WeightsTransposed);

    const std::vector<int32_t> add3BiasShape = {1, 500};
    const float* add3BiasData = reinterpret_cast<float*>(weightsData.get() + byteOffset);
    const uint32_t add3BiasDataLength = product(add3BiasShape) * sizeof(float);
    byteOffset += add3BiasDataLength;
    const webnn::Operand add3BiasConstant =
        utils::BuildConstant(builder, add3BiasShape, add3BiasData, add3BiasDataLength);
    const webnn::Operand add3 = builder.Add(matmul1, add3BiasConstant);

    const webnn::Operand relu = builder.Relu(add3);

    const std::vector<int32_t> newShape2 = {1, -1};
    const webnn::Operand reshape2 = builder.Reshape(relu, newShape2.data(), newShape2.size());

    const std::vector<int32_t> matmul2Shape = {10, 500};
    const float* matmul2Data = reinterpret_cast<float*>(weightsData.get() + byteOffset);
    const uint32_t matmul2DataLength = product(matmul2Shape) * sizeof(float);
    byteOffset += matmul2DataLength;
    const webnn::Operand matmul2Weights =
        utils::BuildConstant(builder, matmul2Shape, matmul2Data, matmul2DataLength);
    const webnn::Operand matmul2WeightsTransposed = builder.Transpose(matmul2Weights);
    const webnn::Operand matmul2 = builder.Matmul(reshape2, matmul2WeightsTransposed);

    const std::vector<int32_t> add4BiasShape = {1, 10};
    const float* add4BiasData = reinterpret_cast<float*>(weightsData.get() + byteOffset);
    const uint32_t add4BiasDataLength = product(add4BiasShape) * sizeof(float);
    byteOffset += add4BiasDataLength;
    const webnn::Operand add4BiasConstant =
        utils::BuildConstant(builder, add4BiasShape, add4BiasData, add4BiasDataLength);
    const webnn::Operand add4 = builder.Add(matmul2, add4BiasConstant);

    const webnn::Operand softmax = builder.Softmax(add4);

    mModel = utils::CreateModel(builder, {{"output", softmax}});
    return true;
}

bool LeNet::Compile(webnn::CompilationOptions const* options) {
    if (!mModel) {
        dawn::ErrorLog() << "Model is not ready.";
        return false;
    }
    const std::chrono::time_point<std::chrono::high_resolution_clock> startTime =
        std::chrono::high_resolution_clock::now();
    mCompilation = utils::AwaitCompile(mModel, options);
    if (!mCompilation) {
        return false;
    }
    const std::chrono::duration<double, std::milli> elapsedTime =
        std::chrono::high_resolution_clock::now() - startTime;
    dawn::InfoLog() << "Compilation Time: " << elapsedTime.count() << " ms";
    return true;
}

webnn::Result LeNet::Compute(const void* inputData, size_t inputLength) {
    if (!mCompilation) {
        dawn::ErrorLog() << "Compilation is not ready.";
        return webnn::Result();
    }
    const std::chrono::time_point<std::chrono::high_resolution_clock> startTime =
        std::chrono::high_resolution_clock::now();
    mResults = utils::AwaitCompute(mCompilation, {{"input", {inputData, inputLength}}});
    if (!mResults) {
        return webnn::Result();
    }
    const std::chrono::duration<double, std::milli> elapsedTime =
        std::chrono::high_resolution_clock::now() - startTime;
    dawn::InfoLog() << "Execution Time: " << elapsedTime.count() << " ms";
    return mResults.Get("output");
}
