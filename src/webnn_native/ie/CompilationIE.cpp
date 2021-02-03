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

#include "webnn_native/ie/CompilationIE.h"

#include <vector>

#include "common/Log.h"
#include "webnn_native/Error.h"
#include "webnn_native/ErrorData.h"
#include "webnn_native/NamedResults.h"
#include "webnn_native/Operand.h"
#include "webnn_native/Result.h"
#include "webnn_native/ie/ErrorIE.h"
#include "webnn_native/ie/ienn_symbol_table/ienn_symbol_table.h"

#define DAWN_CALLBACK_TRY(code, messages)                                     \
    {                                                                         \
        MaybeError maybe_error = CheckStatusCode(code, messages);             \
        if (maybe_error.IsError()) {                                          \
            std::unique_ptr<ErrorData> error = maybe_error.AcquireError();    \
            callback(status, nullptr, error->GetMessage().c_str(), userdata); \
            return;                                                           \
        }                                                                     \
    }                                                                         \
    for (;;)                                                                  \
    break

namespace webnn_native { namespace ie {

    class Result : public ResultBase {
      public:
        using ResultBase::Reference;
        ~Result() {
            ie_compilation_free_buffer(&buffer_);
        }
    };

    Compilation::Compilation(Ref<Model> model) : model_(model) {
    }

    Compilation::~Compilation() {
        IE(ie_compilation_free)(ie_compilation_);
    }

    void Compilation::Compile(WebnnCompileCallback callback,
                              void* userdata,
                              CompilationOptions const* options) {
        // We may leverage https://dawn-review.googlesource.com/c/dawn/+/36360 to
        // implement async compilation as standle-alone component.
        WebnnCompileStatus status = WebnnCompileStatus_Error;
        // Create compilation for IE backend.
        IEStatusCode code =
            IE(ie_create_compilation)(model_->GetInferenceEngineModel(), &ie_compilation_);
        DAWN_CALLBACK_TRY(code, "IE create compilation");
        status = WebnnCompileStatus_Success;
        callback(status, reinterpret_cast<WebnnCompilation>(this), nullptr, userdata);
    }

    void Compilation::ComputeImpl(NamedInputsBase* inputs,
                                  WebnnComputeCallback callback,
                                  void* userdata,
                                  NamedOutputsBase* outputs) {
        WebnnComputeStatus status = WebnnComputeStatus_Error;
        // Set input data to nGraph.
        for (auto& input : inputs->GetRecords()) {
            ie_operand_t ie_operand;
            ie_operand.name = const_cast<char*>(model_->input_id_map_[input.first].c_str());
            IEStatusCode code = IE(ie_compilation_set_input)(
                ie_compilation_, &ie_operand, input.second->buffer, input.second->size);
            DAWN_CALLBACK_TRY(code, "IE set input");
        }

        // Compute the compiled model.
        IEStatusCode code = IE(ie_compilation_compute)(ie_compilation_);
        DAWN_CALLBACK_TRY(code, "IE compute model");
        // Get Data from nGraph with output.
        Ref<NamedResultsBase> results = AcquireRef(new NamedResultsBase());
        size_t output_number = model_->GetOutputsNumber();
        for (size_t i = 0; i < output_number; ++i) {
            std::string output_id = model_->GetOutputId(i);
            void* output_buffer;
            size_t buffer_length;
            IEStatusCode code = IE(ie_compilation_get_buffer)(ie_compilation_, output_id.data(),
                                                              &output_buffer, &buffer_length);
            DAWN_CALLBACK_TRY(code, "IE get buffer");
            ie_dimensions_t ie_dimensions;
            code = IE(ie_compilation_get_dimensions)(ie_compilation_, output_id.data(),
                                                     &ie_dimensions);
            DAWN_CALLBACK_TRY(code, "IE get dimensions");
            std::vector<int32_t> dimensions(ie_dimensions.dims,
                                            ie_dimensions.dims + ie_dimensions.ranks);
            code = IE(ie_compilation_free_dimensions)(&ie_dimensions);
            Ref<ResultBase> result =
                AcquireRef(new Result::ResultBase(output_buffer, buffer_length, dimensions));
            std::string output_name = model_->output_name_map_[output_id];
            results->Set(output_name.c_str(), result.Detach());
            if (outputs != nullptr) {
                const Output* output = outputs->GetRecords().at(output_name);
                ie_operand_t ie_operand;
                ie_operand.name = const_cast<char*>(output_id.c_str());
                IEStatusCode code = IE(ie_compilation_get_output)(ie_compilation_, &ie_operand,
                                                                  output->buffer, output->size);
                DAWN_CALLBACK_TRY(code, "IE get output");
            }
        }
        status = WebnnComputeStatus_Success;
        callback(status, reinterpret_cast<WebnnNamedResults>(results.Detach()), nullptr, userdata);
        return;
    }

}}  // namespace webnn_native::ie
