
#include "dawn_native/ie/compilation_ie.h"

#include <vector>

#include "common/Log.h"
#include "dawn_native/Error.h"
#include "dawn_native/ErrorData.h"
#include "dawn_native/NamedResults.h"
#include "dawn_native/Operand.h"
#include "dawn_native/Result.h"
#include "error_ie.h"
#include "ienn_symbol_table.h"

#define DAWN_CALLBACK_TRY(code, messages)                                      \
  {                                                                            \
    MaybeError maybe_error = CheckStatusCode(code, messages);                  \
    if (maybe_error.IsError()) {                                               \
      std::unique_ptr<ErrorData> error = maybe_error.AcquireError();           \
      callback(status, nullptr, error->GetMessage().c_str(), userdata);        \
      return;                                                                  \
    }                                                                          \
  }                                                                            \
  for (;;)                                                                     \
  break

namespace dawn_native {

namespace ie {

class Result : public ResultBase {
public:
  using ResultBase::Reference;
  ~Result() {
    ie_compilation_free_buffer(&buffer_);
  }
};

Compilation::Compilation(Ref<Model> model) : model_(model) {}

Compilation::~Compilation() { IE(ie_compilation_free)(ie_compilation_); }

MaybeError Compilation::Init(WNNCompileStatus *status) {
  *status = WNNCompileStatus_Error;
  // Create compilation for IE backend.
  IEStatusCode code = IE(ie_create_compilation)(
      model_->GetInferenceEngineModel(), &ie_compilation_);
  DAWN_TRY(CheckStatusCode(code, "IE create compilation"));
  *status = WNNCompileStatus_Success;
  return {};
}

void Compilation::ComputeImpl(NamedInputsBase *inputs,
                              WNNComputeCallback callback, void *userdata,
                              NamedOutputsBase *outputs) {
  WNNComputeStatus status = WNNComputeStatus_Error;
  // Set input data to nGraph.
  for (auto &input : inputs->GetRecords()) {
    ie_operand_t ie_operand;
    ie_operand.name =
        const_cast<char *>(model_->input_id_map_[input.first].c_str());
    IEStatusCode code = IE(ie_compilation_set_input)(
        ie_compilation_, &ie_operand, input.second->buffer, input.second->size);
    DAWN_CALLBACK_TRY(code, "IE set input");
  }

  // Compute the compiled model.
  IEStatusCode code = IE(ie_compilation_compute)(ie_compilation_);
  DAWN_CALLBACK_TRY(code, "IE compute model");

  // Get Data from nGraph with output.
  // TODO(junwei). new memory for output data.
  Ref<NamedResultsBase> results = AcquireRef(new NamedResultsBase());
  size_t output_number = model_->GetOutputsNumber();
  for (size_t i = 0; i < output_number; ++i) {
    std::string output_id = model_->GetOutputId(i);
    void *output_buffer;
    size_t buffer_length;
    IEStatusCode code = IE(ie_compilation_get_buffer)(
        ie_compilation_, output_id.data(), &output_buffer, &buffer_length);
    DAWN_CALLBACK_TRY(code, "IE get buffer");
    // TODO(junwei): get the output dimensions;
    std::vector<int32_t> dimensions;
    Ref<ResultBase> result = AcquireRef(
        new Result::ResultBase(output_buffer, buffer_length, dimensions));
    std::string output_name = model_->output_name_map_[output_id];
    results->Set(output_name.c_str(), result.Detach());
    if (outputs != nullptr) {
      const Output *output = outputs->GetRecords().at(output_name);
      ie_operand_t ie_operand;
      ie_operand.name = const_cast<char *>(output_id.c_str());
      IEStatusCode code = IE(ie_compilation_get_output)(
          ie_compilation_, &ie_operand, output->buffer, output->size);
      DAWN_CALLBACK_TRY(code, "IE get output");
    }
  }
  status = WNNComputeStatus_Success;
  callback(status, reinterpret_cast<WNNNamedResults>(results.Detach()), nullptr,
           userdata);
  return;
}

} // namespace ie

} // namespace dawn_native
