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

#include <memory>
#include <string>

#include <webnn/webnn.h>
#include <webnn/webnn_cpp.h>

class LeNet {
  public:
    LeNet();
    ~LeNet() = default;

    bool Load(const std::string& weigthsPath);
    bool Compile(webnn::CompilationOptions const* options = nullptr);
    webnn::Result Compute(const void* inputData, size_t inputLength);

  private:
    static void UncapturedErrorCallback(WebnnErrorType type, char const* message, void* userData);
    static void ValidationErrorCallback(WebnnErrorType type, char const* message, void* userData);
    static void CompilationCallback(WebnnCompileStatus status,
                                    WebnnCompilation impl,
                                    char const* message,
                                    void* userData);
    static void ComputeCallback(WebnnComputeStatus status,
                                WebnnNamedResults impl,
                                char const* message,
                                void* userData);

    webnn::NeuralNetworkContext mContext;
    webnn::Model mModel;
    webnn::Compilation mCompilation;
    webnn::NamedResults mOutputs;
    // Need to keep the weights data during life cycle of LeNet.
    std::unique_ptr<char> mWeightsData;
    bool mValidationFailed;
};
