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
            free(buffer_);
        }
    };

    Compilation::Compilation(const Ref<Model>& model) : model_(model) {
        std::vector<::dml::Expression> outputs;
        for (auto& output : model_->outputs_) {
            outputs.push_back(output.second);
        }
        // TODO(nhu): investigate other execution flag,
        // e.g. DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION
        compiled_model_.reset(new pydml::CompiledModel(*(model_->graph_),
        // FIXME(nhu): workaround https://github.com/microsoft/DirectML/issues/70
#if defined(_DEBUG)
                                                       DML_EXECUTION_FLAG_DISABLE_META_COMMANDS,
#else
                                                       DML_EXECUTION_FLAG_NONE,
#endif
                                                       outputs));
    }

    void Compilation::ComputeImpl(NamedInputsBase* inputs,
                                  WebnnComputeCallback callback,
                                  void* userdata,
                                  NamedOutputsBase* outputs) {
        // FIXME(nhu): implement async
        for (auto& input : inputs->GetRecords()) {
            ::pydml::Binding* input_binding = model_->inputs_.at(input.first);
            input_binding->data.buffer_ = const_cast<void*>(input.second->buffer);
            input_binding->data.size_ = input.second->size;
            DAWN_DEBUG() << " set input name: " << input.first << ", buffer "
                         << input.second->buffer << ", buffer size: " << input.second->size
                         << ", type: " << DmlTensorDataTypeToString(input_binding->desc.dataType)
                         << ", dimensions: "
                         << DmlTensorDimensionsToString(input_binding->desc.sizes);
        }
        std::vector<pydml::Binding*> input_bindings;
        for (auto& binding : model_->bindings_) {
            input_bindings.push_back(binding.get());
        }
        std::vector<::dml::Expression*> output_expressions;
        std::vector<std::string> output_names;
        if (outputs != nullptr) {
            for (auto& output : outputs->GetRecords()) {
                output_names.push_back(output.first);
                output_expressions.push_back(&(model_->outputs_.at(output.first)));
            }
        } else {
            for (auto& output : model_->outputs_) {
                output_names.push_back(output.first);
                output_expressions.push_back(&(output.second));
            }
        }
        std::vector<pydml::TensorData*> output_tensors =
            model_->device_->Compute(compiled_model_->op.Get(), input_bindings, output_expressions);

        Ref<NamedResultsBase> results = AcquireRef(new NamedResultsBase());
        for (size_t i = 0; i < output_names.size(); ++i) {
            std::string output_name = output_names[i];
            pydml::TensorData* tensor = output_tensors[i];
            void* output_buffer = tensor->Get();
            size_t buffer_length = tensor->Size();
            std::vector<int32_t> dimensions;
            for (auto size : tensor->Desc()->sizes) {
                // convert from uint32_t to int32_t.
                dimensions.push_back(static_cast<int32_t>(size));
            }
            Ref<ResultBase> result =
                AcquireRef(new Result::ResultBase(output_buffer, buffer_length, dimensions));
            results->Set(output_name.c_str(), result.Detach());
            if (outputs != nullptr) {
                const Output* output = outputs->GetRecords().at(output_name);
                if (output->size >= buffer_length) {
                    memcpy(output->buffer, output_buffer, buffer_length);
                }
            }
            DAWN_DEBUG() << " set output name: " << output_name << ", buffer: " << output_buffer
                         << ", buffer size: " << buffer_length
                         << ", type: " << DmlTensorDataTypeToString(tensor->Desc()->dataType)
                         << ", dimensions: " << DmlTensorDimensionsToString(tensor->Desc()->sizes);
            delete tensor;
        }
        WebnnComputeStatus status = WebnnComputeStatus_Success;
        callback(status, reinterpret_cast<WebnnNamedResults>(results.Detach()), nullptr, userdata);
        return;
    }

}}  // namespace webnn_native::dml