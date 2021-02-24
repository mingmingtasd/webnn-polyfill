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
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "MnistUbyte.h"
#include "common/Log.h"
#include "examples/SampleUtils.h"

std::string gImagePath, gModelPath;
void* gInputData;
size_t gInputLength;
const size_t WEIGHTS_LENGTH = 1724336;
const size_t TOP_NUMBER = 3;

void SelectTopKData(std::vector<float> outputData,
                    std::vector<size_t>& topKIndex,
                    std::vector<float>& topKData) {
    std::vector<size_t> indexes(outputData.size());
    std::iota(std::begin(indexes), std::end(indexes), 0);
    std::partial_sort(
        std::begin(indexes), std::begin(indexes) + TOP_NUMBER, std::end(indexes),
        [&outputData](unsigned l, unsigned r) { return outputData[l] > outputData[r]; });
    sort(outputData.rbegin(), outputData.rend());

    for (size_t i = 0; i < TOP_NUMBER; ++i) {
        topKIndex[i] = indexes[i];
        topKData[i] = outputData[i];
    }
}

void PrintResult(webnn::Result output) {
    const float* outputBuffer = static_cast<const float*>(output.Buffer());
    std::vector<float> outputData(outputBuffer, outputBuffer + output.BufferSize() / sizeof(float));
    std::vector<size_t> topKIndex(TOP_NUMBER);
    std::vector<float> topKData(TOP_NUMBER);
    SelectTopKData(outputData, topKIndex, topKData);

    std::cout << std::endl << "Top " << TOP_NUMBER << " results:" << std::endl << std::endl;
    std::cout << "#"
              << "   "
              << "Label"
              << " "
              << "Probability" << std::endl
              << std::endl;
    std::cout.precision(2);
    for (size_t i = 0; i < TOP_NUMBER; ++i) {
        std::cout << i << "   ";
        std::cout << std::left << std::setw(5) << std::fixed << topKIndex[i] << " ";
        std::cout << std::left << std::fixed << 100 * topKData[i] << "%" << std::endl;
    }
}

void ShowUsage() {
    std::cout << std::endl;
    std::cout << "LeNet Example [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                      "
              << "Print a usage message." << std::endl;
    std::cout << "    -i \"<path>\"             "
              << "Required. Path to image." << std::endl;
    std::cout << "    -m \"<path>\"             "
              << "Required. Path to a .bin file with weights for the trained model." << std::endl;
    std::cout << std::endl;
}

bool ParseAndCheckArgs(int argc, const char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp("-h", argv[i]) == 0) {
            return false;
        }
        if (strcmp("-i", argv[i]) == 0 && i + 1 < argc) {
            gImagePath = argv[i + 1];
        } else if (strcmp("-m", argv[i]) == 0 && i + 1 < argc) {
            gModelPath = argv[i + 1];
        }
    }

    bool paramValid = true;
    if (gImagePath.empty()) {
        paramValid = false;
        dawn::ErrorLog() << "Input is required but not set. Please set -i option.";
    }
    if (gModelPath.empty()) {
        paramValid = false;
        dawn::ErrorLog() << "Model is required but not set. Please set -m option.";
    }
    return paramValid;
}

// The Compilation should be released unitl ComputeCallback.
webnn::Compilation gCompilation;
std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
void ComputeCallback(WebnnComputeStatus status,
                     WebnnNamedResults impl,
                     char const* message,
                     void* userData) {
    if (status != WebnnComputeStatus_Success) {
        dawn::ErrorLog() << message;
        return;
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedTime = end - start;
    dawn::InfoLog() << "Compute done, inference time: " << elapsedTime.count() << " ms";

    webnn::NamedResults outputs = outputs.Acquire(impl);
    webnn::Result output = outputs.Get("output");
    PrintResult(output);
}

void CompilationCallback(WebnnCompileStatus status,
                         WebnnCompilation impl,
                         char const* message,
                         void* userData) {
    if (status != WebnnCompileStatus_Success) {
        dawn::ErrorLog() << message;
        return;
    }
    gCompilation = gCompilation.Acquire(impl);
    webnn::Input a;
    a.buffer = gInputData;
    a.size = gInputLength;
    webnn::NamedInputs inputs = CreateCppNamedInputs();
    inputs.Set("input", &a);

    start = std::chrono::high_resolution_clock::now();
    gCompilation.Compute(inputs, ComputeCallback, nullptr, nullptr);
}

int main(int argc, const char* argv[]) {
    DumpMemoryLeaks();
    if (!ParseAndCheckArgs(argc, argv)) {
        ShowUsage();
        return -1;
    }
    webnn::NeuralNetworkContext context = CreateCppNeuralNetworkContext();
    webnn::ModelBuilder builder = context.CreateModelBuilder();

    MnistUbyte reader(gImagePath);
    if (!reader.DataInitialized()) {
        dawn::ErrorLog() << "The input image is invalid.";
        return -1;
    }
    if (reader.Size() != 28 * 28) {
        dawn::ErrorLog() << "The expected size of the input image is 28 * 28, but got "
                         << reader.Size();
        return -1;
    }
    std::vector<float> inputBuffer(reader.GetData().get(), reader.GetData().get() + reader.Size());
    gInputData = inputBuffer.data();
    gInputLength = inputBuffer.size() * sizeof(float);
    FILE* fp = fopen(gModelPath.c_str(), "rb");
    if (!fp) {
        dawn::ErrorLog() << "Failed to open weights file at " << gModelPath;
        return -1;
    }
    void* dataBuffer = malloc(WEIGHTS_LENGTH);
    size_t size = fread(dataBuffer, sizeof(char), WEIGHTS_LENGTH, fp);
    fclose(fp);
    if (size != WEIGHTS_LENGTH) {
        dawn::ErrorLog() << "The expected size of weights file is " << WEIGHTS_LENGTH
                         << ", but got " << size;
        free(dataBuffer);
        return -1;
    }

    uint32_t byteOffset = 0;
    std::vector<int32_t> inputShape = {1, 1, 28, 28};
    webnn::OperandDescriptor inputDesc = {webnn::OperandType::Float32, inputShape.data(),
                                          (uint32_t)inputShape.size()};

    webnn::Operand input = builder.Input("input", &inputDesc);

    std::vector<int32_t> conv2d1FilterShape = {20, 1, 5, 5};
    float* conv2d1FilterData = static_cast<float*>(dataBuffer) + byteOffset;
    byteOffset += product(conv2d1FilterShape);
    webnn::OperandDescriptor conv2d1FilterDesc = {webnn::OperandType::Float32,
                                                  conv2d1FilterShape.data(),
                                                  (uint32_t)conv2d1FilterShape.size()};
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
    webnn::OperandDescriptor conv2d2FilterDesc = {webnn::OperandType::Float32,
                                                  conv2d2FilterShape.data(),
                                                  (uint32_t)conv2d2FilterShape.size()};
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

    free(dataBuffer);
    // Release backend resources in main thread.
    gCompilation = nullptr;
}
