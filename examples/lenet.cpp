// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <memory>
#include <stdlib.h>
#include <vector>
#include <chrono>

#include "SampleUtils.h"
#include "common/Log.h"

std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
void ComputeCallback(WNNComputeStatus status, WNNNamedResults impl,
    char const * message, void* userData) {
  if (status != WNNComputeStatus_Success) {
    dawn::ErrorLog() << message;
    return;
  }
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed_time = end - start;
  dawn::InfoLog() << "inference time: " << elapsed_time.count() << " ms";

  wnn::NamedResults outputs = outputs.Acquire(impl);
  wnn::Result output = outputs.Get("output");

  std::vector<float> expected_data = {0, 0, 0, 1, 0, 0, 0, 0, 0 ,0};
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
  if (expected) {
    dawn::InfoLog() << "The output output as expected.";
  }
}


void CompilationCallback(WNNCompileStatus status, WNNCompilation impl,
    char const * message, void* userData) {
  if (status != WNNCompileStatus_Success) {
    dawn::ErrorLog() << message;
    return;
  }
  wnn::Compilation exe = exe.Acquire(impl);

  std::vector<float> input_buffer = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,3,5,4,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,47,119,210,164,119,116,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,99,233,250,253,252,250,246,72,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,224,253,254,253,254,253,230,72,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,105,240,253,226,253,254,250,136,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,201,251,213,253,254,248,41,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,86,201,251,254,254,249,48,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,35,207,253,254,252,164,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,7,92,139,249,254,254,250,112,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,43,206,245,251,254,254,254,248,43,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,80,239,253,254,254,254,254,251,143,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,111,239,250,250,250,253,252,168,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,74,90,91,98,234,251,170,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,7,134,250,214,34,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,245,253,208,31,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,2,0,0,0,0,0,4,137,251,250,117,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,25,168,91,4,1,0,1,3,34,233,253,245,40,1,0,0,0,0,0,0,0,0,0,0,0,0,0,4,156,247,210,69,39,7,54,93,201,252,250,138,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,60,195,248,248,218,180,235,249,253,251,205,19,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,133,111,247,249,250,253,254,253,226,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,12,115,118,164,245,247,229,57,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,3,4,6,6,6,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  wnn::Input a;
  a.buffer = input_buffer.data();
  a.size = input_buffer.size() * sizeof(float);
  wnn::NamedInputs inputs = CreateCppNamedInputs();
  inputs.Set("input", &a);
  
  start = std::chrono::high_resolution_clock::now();
  exe.Compute(inputs, ComputeCallback, nullptr, nullptr);
}

int main(int argc, const char* argv[]) {
  wnn::NeuralNetworkContext context = CreateCppNeuralNetworkContext();
  wnn::ModelBuilder builder = context.CreateModelBuilder();

  FILE *fp = fopen("lenet.bin", "rb");
  const int32_t length = 1724336;
  void* dataBuffer = malloc(length);
  size_t size = fread(dataBuffer, sizeof(char), length, fp);
  dawn::InfoLog() << "size: " << size;
  fclose(fp);

  uint32_t byteOffset = 0;
  std::vector<int32_t> inputShape = {1, 1, 28, 28};
  wnn::OperandDescriptor inputDesc = {wnn::OperandType::Float32, inputShape.data(),
                                             (uint32_t)inputShape.size()};

  wnn::Operand input = builder.Input("input", &inputDesc);

  std::vector<int32_t> conv2d1FilterShape = {20, 1, 5, 5};
  float* conv2d1FilterData = static_cast<float*>(dataBuffer) + byteOffset;
  byteOffset += product(conv2d1FilterShape);
  wnn::OperandDescriptor conv2d1FilterDesc = {wnn::OperandType::Float32, conv2d1FilterShape.data(),
                                             (uint32_t)conv2d1FilterShape.size()};
  wnn::Operand conv2d1FilterConstant = builder.Constant(&conv2d1FilterDesc, conv2d1FilterData, 
		                       product(conv2d1FilterShape) * sizeof(float));
  wnn::Conv2dOptions conv2d1Options = {};
  wnn::Operand conv1 = builder.Conv2d(input, conv2d1FilterConstant, &conv2d1Options);

  std::vector<int32_t> add1BiasShape = {1, 20, 1, 1};
  float* add1BiasData = static_cast<float*>(dataBuffer) + byteOffset;
  byteOffset += product(add1BiasShape);
  wnn::OperandDescriptor add1BiasDesc = {wnn::OperandType::Float32, add1BiasShape.data(),
                                             (uint32_t)add1BiasShape.size()};
  wnn::Operand add1BiasConstant = builder.Constant(&add1BiasDesc, add1BiasData,
                                       product(add1BiasShape) * sizeof(float));
  wnn::Operand add1 = builder.Add(conv1, add1BiasConstant);

  wnn::Pool2dOptions options = {};
  std::vector<int32_t> windowDimensions = {2, 2};
  options.windowDimensions = windowDimensions.data();
  std::vector<int32_t> strides = {2, 2};
  options.strides = strides.data();
  options.stridesCount = 2;
  std::vector<int32_t> padding = {0, 0, 0, 0};
  options.padding = padding.data();
  options.paddingCount = 4;
  wnn::Operand pool1 = builder.MaxPool2d(add1, &options);

  std::vector<int32_t> conv2d2FilterShape = {50, 20, 5, 5,};
  float* conv2d2FilterData = static_cast<float*>(dataBuffer) + byteOffset;
  byteOffset += product(conv2d2FilterShape);
  wnn::OperandDescriptor conv2d2FilterDesc = {wnn::OperandType::Float32, conv2d2FilterShape.data(),
                                             (uint32_t)conv2d2FilterShape.size()};
  wnn::Operand conv2d2FilterConstant = builder.Constant(&conv2d2FilterDesc, conv2d2FilterData,
                                       product(conv2d2FilterShape) * sizeof(float));
  wnn::Conv2dOptions conv2d2Options = {};
  wnn::Operand conv2 = builder.Conv2d(pool1, conv2d2FilterConstant, &conv2d2Options);

  std::vector<int32_t> add2BiasShape = {1, 50, 1, 1};
  float* add2BiasData = static_cast<float*>(dataBuffer) + byteOffset;
  byteOffset += product(add2BiasShape);
  wnn::OperandDescriptor add2BiasDesc = {wnn::OperandType::Float32, add2BiasShape.data(),
                                             (uint32_t)add2BiasShape.size()};
  wnn::Operand add2BiasConstant = builder.Constant(&add2BiasDesc, add2BiasData,
                                       product(add2BiasShape) * sizeof(float));
  wnn::Operand add2 = builder.Add(conv2, add2BiasConstant);

  wnn::Pool2dOptions options2 = {};
  std::vector<int32_t> windowDimensions2 = {2, 2};
  options2.windowDimensions = windowDimensions2.data();
  std::vector<int32_t> strides2 = {2, 2};
  options2.strides = strides2.data();
  options2.stridesCount = 2;
  std::vector<int32_t> padding2 = {0, 0, 0, 0};
  options2.padding = padding2.data();
  options2.paddingCount = 4;
  wnn::Operand pool2 = builder.MaxPool2d(add2, &options2);

  std::vector<int32_t> newShape = {1, -1};
  wnn::Operand  reshape1 = builder.Reshape(pool2, newShape.data(), newShape.size());
  // skip the new shape
  byteOffset += 4;

  std::vector<int32_t> matmulShape = {500, 800};
  float* matmulData = static_cast<float*>(dataBuffer) + byteOffset;
  byteOffset +=  product(matmulShape);
  wnn::OperandDescriptor matmulDataDesc = {wnn::OperandType::Float32, matmulShape.data(),
                    (uint32_t)matmulShape.size()};
  wnn::Operand matmulWeights = builder.Constant(&matmulDataDesc,
                                 matmulData, product(matmulShape) * sizeof(float));

  wnn::Operand matmul1WeightsTransposed = builder.Transpose(matmulWeights);
  wnn::Operand matmul1 = builder.Matmul(reshape1, matmul1WeightsTransposed);

  std::vector<int32_t> add3BiasShape = {1, 500};
  float* add3BiasData = static_cast<float*>(dataBuffer) + byteOffset;
  byteOffset +=  product(add3BiasShape);
  wnn::OperandDescriptor add3BiasDesc = {wnn::OperandType::Float32, add3BiasShape.data(),
                                             (uint32_t)add3BiasShape.size()};
  wnn::Operand add3BiasConstant = builder.Constant(&add3BiasDesc, add3BiasData,
                                       product(add3BiasShape) * sizeof(float));
  wnn::Operand add3 = builder.Add(matmul1, add3BiasConstant);

  wnn::Operand relu = builder.Relu(add3);
  
  std::vector<int32_t> newShape2 = {1, -1};
  wnn::Operand  reshape2 = builder.Reshape(relu, newShape2.data(), newShape2.size());

  std::vector<int32_t> matmulShape2 = {10, 500};
  float* matmulData2 = static_cast<float*>(dataBuffer) + byteOffset;
  byteOffset +=  product(matmulShape2);
  wnn::OperandDescriptor matmulData2Desc = {wnn::OperandType::Float32, matmulShape2.data(),
                    (uint32_t)matmulShape2.size()};
  wnn::Operand matmulWeights2 = builder.Constant(&matmulData2Desc,
                                 matmulData2, product(matmulShape2) * sizeof(float));

  wnn::Operand matmul1WeightsTransposed2 = builder.Transpose(matmulWeights2);
  wnn::Operand matmul2 = builder.Matmul(reshape2, matmul1WeightsTransposed2);

  std::vector<int32_t> add4BiasShape = {1, 10};
  float* add4BiasData = static_cast<float*>(dataBuffer) + byteOffset;
  byteOffset += product(add4BiasShape);
  wnn::OperandDescriptor add4BiasDesc = {wnn::OperandType::Float32, add4BiasShape.data(),
                                             (uint32_t)add4BiasShape.size()};
  wnn::Operand add4BiasConstant = builder.Constant(&add4BiasDesc, add4BiasData,
                                       product(add4BiasShape) * sizeof(float));
  wnn::Operand add4 = builder.Add(matmul2, add4BiasConstant);

  wnn::Operand softmax = builder.Softmax(add4);

  wnn::NamedOperands named_operands = CreateCppNamedOperands();
  named_operands.Set("output", softmax);
  wnn::Model model = builder.CreateModel(named_operands);
  model.Compile(CompilationCallback, nullptr);

  free(dataBuffer);
}

