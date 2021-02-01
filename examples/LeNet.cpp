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

#include <stdlib.h>
#include <chrono>
#include <memory>
#include <vector>

#include "SampleUtils.h"
#include "common/Log.h"

// The Compilation should be released unitl ComputeCallback.
webnn::Compilation gCompilation;
utils::ComputeSync gComputeSync;
std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
void ComputeCallback(WEBNNComputeStatus status,
                     WEBNNNamedResults impl,
                     char const* message,
                     void* userData) {
    if (status != WEBNNComputeStatus_Success) {
        dawn::ErrorLog() << message;
        gComputeSync.Finish();
        return;
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedTime = end - start;
    dawn::InfoLog() << "inference time: " << elapsedTime.count() << " ms";

    webnn::NamedResults outputs = outputs.Acquire(impl);
    webnn::Result output = outputs.Get("output");

    std::vector<float> expectedData = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
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
    if (expected) {
        dawn::InfoLog() << "The output output as expected.";
    }
    gComputeSync.Finish();
}

void CompilationCallback(WEBNNCompileStatus status,
                         WEBNNCompilation impl,
                         char const* message,
                         void* userData) {
    if (status != WEBNNCompileStatus_Success) {
        dawn::ErrorLog() << message;
        gComputeSync.Finish();
        return;
    }
    gCompilation = gCompilation.Acquire(impl);

    std::vector<float> inputBuffer = {
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   1,   3,   5,   4,   3,   3,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   4,   47,  119, 210,
        164, 119, 116, 10,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   2,   99,  233, 250, 253, 252, 250, 246, 72,  3,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   24,  224,
        253, 254, 253, 254, 253, 230, 72,  2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   3,   105, 240, 253, 226, 253, 254, 250, 136, 3,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   8,   201, 251, 213, 253, 254, 248, 41,  1,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   2,   86,  201, 251, 254, 254, 249,
        48,  1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   5,   35,  207, 253, 254, 252, 164, 4,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   7,   92,  139, 249, 254,
        254, 250, 112, 3,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   1,   43,  206, 245, 251, 254, 254, 254, 248, 43,  1,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   2,   80,  239, 253, 254,
        254, 254, 254, 251, 143, 3,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   5,   111, 239, 250, 250, 250, 253, 252, 168, 4,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   4,
        74,  90,  91,  98,  234, 251, 170, 5,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   2,   2,   2,   7,   134, 250, 214, 34,
        1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   16,  245, 253, 208, 31,  0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   4,   2,   0,   0,   0,   0,   0,   4,   137, 251, 250,
        117, 3,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   25,  168,
        91,  4,   1,   0,   1,   3,   34,  233, 253, 245, 40,  1,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   4,   156, 247, 210, 69,  39,  7,   54,  93,  201, 252,
        250, 138, 11,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,
        60,  195, 248, 248, 218, 180, 235, 249, 253, 251, 205, 19,  0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   5,   133, 111, 247, 249, 250, 253, 254,
        253, 226, 11,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   3,   12,  115, 118, 164, 245, 247, 229, 57,  1,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,   3,   4,
        6,   6,   6,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0};

    webnn::Input a;
    a.buffer = inputBuffer.data();
    a.size = inputBuffer.size() * sizeof(float);
    webnn::NamedInputs inputs = CreateCppNamedInputs();
    inputs.Set("input", &a);

    start = std::chrono::high_resolution_clock::now();
    gCompilation.Compute(inputs, ComputeCallback, nullptr, nullptr);
}

int main(int argc, const char* argv[]) {
    webnn::NeuralNetworkContext context = CreateCppNeuralNetworkContext();
    webnn::ModelBuilder builder = context.CreateModelBuilder();

    FILE* fp = fopen("lenet.bin", "rb");
    const int32_t length = 1724336;
    void* dataBuffer = malloc(length);
    size_t size = fread(dataBuffer, sizeof(char), length, fp);
    dawn::InfoLog() << "size: " << size;
    fclose(fp);

    uint32_t byteOffset = 0;
    std::vector<int32_t> inputShape = {1, 1, 28, 28};
    webnn::OperandDescriptor inputDesc = {webnn::OperandType::Float32, inputShape.data(),
                                        (uint32_t)inputShape.size()};

    webnn::Operand input = builder.Input("input", &inputDesc);

    std::vector<int32_t> conv2d1FilterShape = {20, 1, 5, 5};
    float* conv2d1FilterData = static_cast<float*>(dataBuffer) + byteOffset;
    byteOffset += product(conv2d1FilterShape);
    webnn::OperandDescriptor conv2d1FilterDesc = {
        webnn::OperandType::Float32, conv2d1FilterShape.data(), (uint32_t)conv2d1FilterShape.size()};
    webnn::Operand conv2d1FilterConstant = builder.Constant(
        &conv2d1FilterDesc, conv2d1FilterData, product(conv2d1FilterShape) * sizeof(float));
    webnn::Conv2dOptions conv2d1Options = {};
    webnn::Operand conv1 = builder.Conv2d(input, conv2d1FilterConstant, &conv2d1Options);

    std::vector<int32_t> add1BiasShape = {1, 20, 1, 1};
    float* add1BiasData = static_cast<float*>(dataBuffer) + byteOffset;
    byteOffset += product(add1BiasShape);
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

    std::vector<int32_t> conv2d2FilterShape = {
        50,
        20,
        5,
        5,
    };
    float* conv2d2FilterData = static_cast<float*>(dataBuffer) + byteOffset;
    byteOffset += product(conv2d2FilterShape);
    webnn::OperandDescriptor conv2d2FilterDesc = {
        webnn::OperandType::Float32, conv2d2FilterShape.data(), (uint32_t)conv2d2FilterShape.size()};
    webnn::Operand conv2d2FilterConstant = builder.Constant(
        &conv2d2FilterDesc, conv2d2FilterData, product(conv2d2FilterShape) * sizeof(float));
    webnn::Conv2dOptions conv2d2Options = {};
    webnn::Operand conv2 = builder.Conv2d(pool1, conv2d2FilterConstant, &conv2d2Options);

    std::vector<int32_t> add2BiasShape = {1, 50, 1, 1};
    float* add2BiasData = static_cast<float*>(dataBuffer) + byteOffset;
    byteOffset += product(add2BiasShape);
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
    // skip the new shape
    byteOffset += 4;

    std::vector<int32_t> matmulShape = {500, 800};
    float* matmulData = static_cast<float*>(dataBuffer) + byteOffset;
    byteOffset += product(matmulShape);
    webnn::OperandDescriptor matmulDataDesc = {webnn::OperandType::Float32, matmulShape.data(),
                                             (uint32_t)matmulShape.size()};
    webnn::Operand matmulWeights =
        builder.Constant(&matmulDataDesc, matmulData, product(matmulShape) * sizeof(float));

    webnn::Operand matmul1WeightsTransposed = builder.Transpose(matmulWeights);
    webnn::Operand matmul1 = builder.Matmul(reshape1, matmul1WeightsTransposed);

    std::vector<int32_t> add3BiasShape = {1, 500};
    float* add3BiasData = static_cast<float*>(dataBuffer) + byteOffset;
    byteOffset += product(add3BiasShape);
    webnn::OperandDescriptor add3BiasDesc = {webnn::OperandType::Float32, add3BiasShape.data(),
                                           (uint32_t)add3BiasShape.size()};
    webnn::Operand add3BiasConstant =
        builder.Constant(&add3BiasDesc, add3BiasData, product(add3BiasShape) * sizeof(float));
    webnn::Operand add3 = builder.Add(matmul1, add3BiasConstant);

    webnn::Operand relu = builder.Relu(add3);

    std::vector<int32_t> newShape2 = {1, -1};
    webnn::Operand reshape2 = builder.Reshape(relu, newShape2.data(), newShape2.size());

    std::vector<int32_t> matmulShape2 = {10, 500};
    float* matmulData2 = static_cast<float*>(dataBuffer) + byteOffset;
    byteOffset += product(matmulShape2);
    webnn::OperandDescriptor matmulData2Desc = {webnn::OperandType::Float32, matmulShape2.data(),
                                              (uint32_t)matmulShape2.size()};
    webnn::Operand matmulWeights2 =
        builder.Constant(&matmulData2Desc, matmulData2, product(matmulShape2) * sizeof(float));

    webnn::Operand matmul1WeightsTransposed2 = builder.Transpose(matmulWeights2);
    webnn::Operand matmul2 = builder.Matmul(reshape2, matmul1WeightsTransposed2);

    std::vector<int32_t> add4BiasShape = {1, 10};
    float* add4BiasData = static_cast<float*>(dataBuffer) + byteOffset;
    byteOffset += product(add4BiasShape);
    webnn::OperandDescriptor add4BiasDesc = {webnn::OperandType::Float32, add4BiasShape.data(),
                                           (uint32_t)add4BiasShape.size()};
    webnn::Operand add4BiasConstant =
        builder.Constant(&add4BiasDesc, add4BiasData, product(add4BiasShape) * sizeof(float));
    webnn::Operand add4 = builder.Add(matmul2, add4BiasConstant);

    webnn::Operand softmax = builder.Softmax(add4);

    webnn::NamedOperands namedOperands = CreateCppNamedOperands();
    namedOperands.Set("output", softmax);
    webnn::Model model = builder.CreateModel(namedOperands);
    model.Compile(CompilationCallback, nullptr);

    gComputeSync.Wait();

    free(dataBuffer);
    // Release backend resources in main thread.
    gCompilation = nullptr;
}
