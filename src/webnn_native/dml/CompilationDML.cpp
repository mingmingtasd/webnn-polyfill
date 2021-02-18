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

#include "webnn_native/dml/CompilationDML.h"

#include <vector>

#include "common/Log.h"
#include "webnn_native/NamedResults.h"
#include "webnn_native/Operand.h"
#include "webnn_native/Result.h"
#include "webnn_native/dml/deps/src/precomp.h"

namespace webnn_native { namespace dml {

    class Result : public ResultBase {
      public:
        using ResultBase::Reference;
        ~Result() {
            free(mBuffer);
        }
    };

    Compilation::Compilation(const Ref<Model>& model) : mModel(model) {
        std::vector<::dml::Expression> outputs;
        for (auto& output : mModel->mOutputs) {
            outputs.push_back(output.second);
        }
        // TODO(nhu): investigate other execution flag,
        // e.g. DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION
        mCompiledModel.reset(
            new pydml::CompiledModel(*(mModel->mGraph), DML_EXECUTION_FLAG_NONE, outputs));
    }

    void Compilation::ComputeImpl(NamedInputsBase* inputs,
                                  WebnnComputeCallback callback,
                                  void* userdata,
                                  NamedOutputsBase* outputs) {
        // FIXME(nhu): implement async
        for (auto& input : inputs->GetRecords()) {
            ::pydml::Binding* inputBinding = mModel->mInputs.at(input.first);
            inputBinding->data.buffer = const_cast<void*>(input.second->buffer);
            inputBinding->data.size = input.second->size;
            DAWN_DEBUG() << " set input name: " << input.first << ", buffer "
                         << input.second->buffer << ", buffer size: " << input.second->size
                         << ", type: " << DmlTensorDataTypeToString(inputBinding->desc.dataType)
                         << ", dimensions: "
                         << DmlTensorDimensionsToString(inputBinding->desc.sizes);
        }
        std::vector<pydml::Binding*> inputBindings;
        for (auto& binding : mModel->mBindings) {
            inputBindings.push_back(binding.get());
        }
        std::vector<::dml::Expression*> outputExpressions;
        std::vector<std::string> outputNames;
        if (outputs != nullptr) {
            for (auto& output : outputs->GetRecords()) {
                outputNames.push_back(output.first);
                outputExpressions.push_back(&(mModel->mOutputs.at(output.first)));
            }
        } else {
            for (auto& output : mModel->mOutputs) {
                outputNames.push_back(output.first);
                outputExpressions.push_back(&(output.second));
            }
        }
        std::vector<pydml::TensorData*> outputTensors =
            mModel->mDevice->Compute(mCompiledModel->op.Get(), inputBindings, outputExpressions);

        Ref<NamedResultsBase> results = AcquireRef(new NamedResultsBase());
        for (size_t i = 0; i < outputNames.size(); ++i) {
            std::string outputName = outputNames[i];
            pydml::TensorData* tensor = outputTensors[i];
            void* outputBuffer = tensor->Get();
            size_t bufferLength = tensor->Size();
            std::vector<int32_t> dimensions;
            for (auto size : tensor->Desc()->sizes) {
                // convert from uint32_t to int32_t.
                dimensions.push_back(static_cast<int32_t>(size));
            }
            Ref<ResultBase> result =
                AcquireRef(new Result::ResultBase(outputBuffer, bufferLength, dimensions));
            results->Set(outputName.c_str(), result.Detach());
            if (outputs != nullptr) {
                const Output* output = outputs->GetRecords().at(outputName);
                if (output->size >= bufferLength) {
                    memcpy(output->buffer, outputBuffer, bufferLength);
                }
            }
            DAWN_DEBUG() << " set output name: " << outputName << ", buffer: " << outputBuffer
                         << ", buffer size: " << bufferLength
                         << ", type: " << DmlTensorDataTypeToString(tensor->Desc()->dataType)
                         << ", dimensions: " << DmlTensorDimensionsToString(tensor->Desc()->sizes);
            delete tensor;
        }
        WebnnComputeStatus status = WebnnComputeStatus_Success;
        callback(status, reinterpret_cast<WebnnNamedResults>(results.Detach()), nullptr, userdata);
        return;
    }

}}  // namespace webnn_native::dml
